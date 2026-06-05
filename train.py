import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
# from pytorch_msssim import ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel_Xray
from refinement import PriorRefinementController
from losses import LowResPriorOccupancyLoss, masked_sobel_edge_loss, sobel_edge_loss
from utils.general_utils import safe_state, gen_log
from tqdm import tqdm
from utils.image_utils import psnr, time2file_name
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import datetime
import time
import yaml
from utils.analysis_utils import (
    append_jsonl,
    export_gaussian_statistics,
    save_dataset_split,
    save_init_statistics,
    save_json,
    save_render_snapshot,
)

from pdb import set_trace as stx


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    render_iterations,
    analysis_iterations,
):
    first_iter = 0
    exp_logger = prepare_output_and_logger(dataset)
    exp_logger.info("Training parameters: {}".format(vars(opt)))

    gaussians = GaussianModel_Xray(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    save_dataset_split(dataset.model_path, scene)
    gaussians.training_setup(opt)
    refinement_controller = None
    prior_analysis_controller = None
    occupancy_loss = None
    if getattr(opt, "use_prior_refinement", False):
        refinement_prior_path = getattr(opt, "refinement_prior_path", "") or getattr(dataset, "prior_path", "")
        if scene.volume_positions is None:
            raise ValueError("Prior refinement requires X-ray volume positions from the scene")
        refinement_controller = PriorRefinementController(
            prior_path=refinement_prior_path,
            volume_positions=scene.volume_positions,
            roi_densify_weight=getattr(opt, "roi_densify_weight", 2.0),
            empty_prune_opacity=getattr(opt, "empty_prune_opacity", 0.01),
            empty_prune_mask_source=getattr(opt, "empty_prune_mask_source", "roi"),
        )
        prior_analysis_controller = refinement_controller
        refinement_description = refinement_controller.save_description(dataset.model_path)
        exp_logger.info(f"Enabled prior refinement: {refinement_description}")
    elif getattr(dataset, "prior_path", ""):
        if scene.volume_positions is None:
            raise ValueError("Prior allocation analysis requires X-ray volume positions from the scene")
        prior_analysis_controller = PriorRefinementController(
            prior_path=getattr(dataset, "prior_path", ""),
            volume_positions=scene.volume_positions,
            roi_densify_weight=getattr(opt, "roi_densify_weight", 2.0),
            empty_prune_opacity=getattr(opt, "empty_prune_opacity", 0.01),
            empty_prune_mask_source=getattr(opt, "empty_prune_mask_source", "roi"),
        )
        analysis_description = prior_analysis_controller.save_description(dataset.model_path)
        exp_logger.info(f"Enabled prior allocation analysis: {analysis_description}")
    if getattr(opt, "use_occ_loss", False):
        occ_prior_path = getattr(opt, "refinement_prior_path", "") or getattr(dataset, "prior_path", "")
        if not occ_prior_path:
            raise ValueError("use_occ_loss requires refinement_prior_path or prior_path")
        occupancy_loss = LowResPriorOccupancyLoss(
            prior_path=occ_prior_path,
            volume_positions=scene.volume_positions,
            grid_size=getattr(opt, "occ_grid_size", 32),
            source=getattr(opt, "occ_source", "roi"),
            loss_type=getattr(opt, "occ_loss_type", "l1"),
        )
        exp_logger.info(
            f"Enabled occupancy consistency loss: prior={occ_prior_path}, "
            f"source={getattr(opt, 'occ_source', 'roi')}, grid={getattr(opt, 'occ_grid_size', 32)}, "
            f"type={getattr(opt, 'occ_loss_type', 'l1')}, lambda={getattr(opt, 'lambda_occ', 0.001)}"
        )
    if getattr(dataset, "save_init_analysis", False):
        save_init_statistics(dataset.model_path, scene.init_info, gaussians)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    training_wall_start = time.time()

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.normalized_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        ssim_term = 1.0 - ssim(image, gt_image)
        base_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_term
        edge_loss_value = torch.zeros((), dtype=image.dtype, device=image.device)
        if getattr(opt, "use_edge_loss", False):
            edge_mask_mode = getattr(opt, "edge_loss_mask_mode", "none")
            if edge_mask_mode == "target_edge":
                edge_loss_value = masked_sobel_edge_loss(
                    image,
                    gt_image,
                    quantile=getattr(opt, "edge_mask_quantile", 0.80),
                    dilation=getattr(opt, "edge_mask_dilation", 1),
                )
            elif edge_mask_mode == "none":
                edge_loss_value = sobel_edge_loss(image, gt_image)
            else:
                raise ValueError(f"Unsupported edge_loss_mask_mode: {edge_mask_mode}")
        occ_loss_value = torch.zeros((), dtype=image.dtype, device=image.device)
        occ_stats = {}
        occ_gaussian_grid = None
        occ_prior_grid = None
        if occupancy_loss is not None and iteration >= getattr(opt, "occ_warmup_from_iter", 0):
            occ_loss_value, occ_stats, occ_gaussian_grid, occ_prior_grid = occupancy_loss(gaussians)
        loss = (
            base_loss
            + getattr(opt, "edge_loss_weight", 0.0) * edge_loss_value
            + getattr(opt, "lambda_occ", 0.0) * occ_loss_value
        )
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == 1 or iteration % 100 == 0 or iteration in testing_iterations:
                append_jsonl(
                    os.path.join(dataset.model_path, "loss_history.jsonl"),
                    {
                        "iteration": int(iteration),
                        "l1_loss": float(Ll1.item()),
                        "ssim_loss": float(ssim_term.item()),
                        "base_loss": float(base_loss.item()),
                        "edge_loss": float(edge_loss_value.item()),
                        "edge_loss_weight": float(getattr(opt, "edge_loss_weight", 0.0)),
                        "edge_loss_mask_mode": getattr(opt, "edge_loss_mask_mode", "none"),
                        "edge_mask_quantile": float(getattr(opt, "edge_mask_quantile", 0.80)),
                        "edge_mask_dilation": int(getattr(opt, "edge_mask_dilation", 1)),
                        "occ_loss": float(occ_loss_value.item()),
                        "lambda_occ": float(getattr(opt, "lambda_occ", 0.0)),
                        "use_occ_loss": bool(getattr(opt, "use_occ_loss", False)),
                        **occ_stats,
                        "total_loss": float(loss.item()),
                    },
                )
            if (
                occupancy_loss is not None
                and occ_gaussian_grid is not None
                and getattr(opt, "occ_debug_interval", 0) > 0
                and iteration % getattr(opt, "occ_debug_interval", 1000) == 0
            ):
                occupancy_loss.save_debug(
                    occ_gaussian_grid,
                    occ_prior_grid,
                    os.path.join(dataset.model_path, "occupancy_debug", f"iteration_{iteration}"),
                )
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                exp_logger,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                render_iterations,
                analysis_iterations,
                scene,
                render,
                (pipe, background),
                dataset,
                prior_analysis_controller,
            )

            if iteration in saving_iterations:
                exp_logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    grad_multipliers = None
                    extra_prune_mask = None
                    if refinement_controller is not None and iteration >= getattr(opt, "refinement_from_iter", 0):
                        refinement_grad_multipliers, refinement_extra_prune_mask, refinement_stats = refinement_controller.build_iteration_bias(gaussians)
                        if getattr(opt, "roi_densify_enabled", False):
                            grad_multipliers = refinement_grad_multipliers
                        if getattr(opt, "empty_prune_enabled", False):
                            extra_prune_mask = refinement_extra_prune_mask
                        append_jsonl(
                            os.path.join(dataset.model_path, "refinement", "refinement_history.jsonl"),
                            {
                                "iteration": int(iteration),
                                "roi_densify_enabled": bool(getattr(opt, "roi_densify_enabled", False)),
                                "empty_prune_enabled": bool(getattr(opt, "empty_prune_enabled", False)),
                                **refinement_stats,
                            },
                        )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        grad_multipliers=grad_multipliers,
                        extra_prune_mask=extra_prune_mask,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    total_training_time = time.time() - training_wall_start
    exp_logger.info(f"Training completed in {total_training_time:.4f} s")
    save_json(
        os.path.join(dataset.model_path, "training_summary.json"),
        {
            "scene": dataset.scene,
            "iterations": int(opt.iterations),
            "total_training_time_sec": total_training_time,
            "train_views": len(scene.getTrainCameras()),
            "test_views": len(scene.getTestCameras()),
            "render_iterations": sorted(render_iterations),
            "analysis_iterations": sorted(analysis_iterations),
            "testing_iterations": sorted(testing_iterations),
            "saving_iterations": sorted(saving_iterations),
        },
    )


def prepare_output_and_logger(args):
    if not args.model_path:
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
        args.model_path = os.path.join("./output/", args.scene, date_time)

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    exp_logger = gen_log(args.model_path)

    return exp_logger


def training_report(
    exp_logger,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    render_iterations,
    analysis_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    args,
    prior_analysis_controller=None,
):
    if exp_logger and (iteration == 0 or (iteration + 1) % 100 == 0):
        exp_logger.info(
            f"Iter:{iteration}, L1 loss={Ll1.item():.4g}, Total loss={loss.item():.4g}, Time:{int(elapsed)}")

    if iteration in analysis_iterations:
        allocation_stats = None
        if prior_analysis_controller is not None:
            allocation_stats = prior_analysis_controller.allocation_statistics(scene.gaussians, iteration=iteration)
        export_gaussian_statistics(
            args.model_path,
            iteration,
            scene.gaussians,
            bins=args.analysis_bins,
            allocation_stats=allocation_stats,
        )
        exp_logger.info(f"[ITER {iteration}] Exported Gaussian statistics")

    if iteration in render_iterations:
        if len(scene.getTestCameras()) > 0:
            save_render_snapshot(
                args.model_path,
                iteration,
                "test",
                scene.getTestCameras(),
                scene.gaussians,
                renderFunc,
                renderArgs,
                max_views=args.render_num_views,
            )
        if len(scene.getTrainCameras()) > 0:
            save_render_snapshot(
                args.model_path,
                iteration,
                "train",
                scene.getTrainCameras(),
                scene.gaussians,
                renderFunc,
                renderArgs,
                max_views=args.render_num_views,
            )
        exp_logger.info(f"[ITER {iteration}] Saved render snapshots")

    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0 and config['name'] == 'test':
                psnr_test = 0.0
                ssim_test = 0.0
                start = time.time()
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    image_backnorm = (viewpoint.max_value - viewpoint.min_value) * image + viewpoint.min_value

                    image = image.mean(dim=0, keepdim=True)
                    image_backnorm = image_backnorm.mean(dim=0, keepdim=True)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_image_norm = viewpoint.normalized_image.to("cuda")

                    ssim_test += ssim(image_backnorm, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image_norm).mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])

                end = time.time()
                fps = len(config['cameras']) / (end - start)
                exp_logger.info(f"Testing Speed: {len(config['cameras']) / (end - start)} fps")
                exp_logger.info(f"Testing Time: {end - start} s")
                exp_logger.info(
                    "\n[ITER {}] Evaluating {}: SSIM = {}, PSNR = {}".format(iteration, config['name'], ssim_test,
                                                                             psnr_test))
                append_jsonl(
                    os.path.join(args.model_path, "metrics_history.jsonl"),
                    {
                        "iteration": int(iteration),
                        "split": config["name"],
                        "ssim": float(ssim_test.item()),
                        "psnr": float(psnr_test.item()),
                        "fps": float(fps),
                        "eval_time_sec": float(end - start),
                        "num_views": int(len(config["cameras"])),
                        "num_gaussians": int(scene.gaussians.get_xyz.shape[0]),
                    },
                )

        if exp_logger:
            exp_logger.info(f'Iter:{iteration}, total_points:{scene.gaussians.get_xyz.shape[0]}')
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)  #
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/chest.yaml', help='Path to the configuration file')
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[100, 2_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000, ])
    parser.add_argument("--render_iterations", nargs="+", type=int, default=[100, 2_000, 20_000])
    parser.add_argument("--analysis_iterations", nargs="+", type=int, default=[100, 2_000, 20_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu_id", default="0", help="gpu to use")
    args = parser.parse_args(sys.argv[1:])

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)

    args.test_iterations = sorted(set(args.test_iterations))
    args.render_iterations = sorted(set(args.render_iterations))
    args.analysis_iterations = sorted(set(args.analysis_iterations))
    args.save_iterations = sorted(set(list(args.save_iterations) + [args.iterations]))

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.render_iterations,
        args.analysis_iterations,
    )

    print("\nTraining complete.")
