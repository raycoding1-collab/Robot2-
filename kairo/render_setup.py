"""Cycles AgX render presets and Blender 5.0 optical cinema compositor.

Presets:
  - motion   : 960x540,  12 samples (fast GI) -> quick motion verification playblast
  - lookdev  : 1280x720, 48 samples (full GI) -> fast still look-dev iteration
  - hero     : 1920x1080, 128 samples (full GI, adaptive 0.015) -> broadcast 1080p quality
  - final    : 1920x1080, 256 samples (full GI, adaptive 0.010) -> cinema 1080p master
  - cinema4k : 3840x2160, 512 samples (full GI, adaptive 0.008) -> GPU 4K master
"""
from __future__ import annotations

import bpy

PRESETS = {
    # name: (width, height, samples, adaptive_threshold, fast_gi)
    "motion":       (960,  540,  10, 0.065, True),
    "preview":      (1280, 720,  20, 0.035, True),
    "fast_lookdev": (960,  540,  28, 0.032, False),
    "lookdev":      (1280, 720,  48, 0.025, False),
    "hd_preview":   (1920, 1080, 32, 0.028, True),
    "hd_fast":      (1920, 1080, 64, 0.022, False),
    "hero":         (1920, 1080, 128, 0.015, False),
    "final":        (1920, 1080, 256, 0.010, False),
    "cinema4k":     (3840, 2160, 512, 0.008, False),
}


def configure_render(preset: str = "final", use_gpu: bool = False, frame_end: int | None = None):
    sc = bpy.context.scene
    w, h, samples, thr, fast = PRESETS[preset]

    r = sc.render
    r.engine = "CYCLES"
    r.resolution_x = w
    r.resolution_y = h
    r.resolution_percentage = 100
    r.fps = 24
    r.fps_base = 1.0
    r.film_transparent = False
    r.use_motion_blur = True
    r.motion_blur_shutter = 0.50  # 180-degree cinema shutter
    r.motion_blur_position = "CENTER"
    r.use_persistent_data = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_depth = "8"
    r.image_settings.color_mode = "RGB"

    cy = sc.cycles
    cy.device = "GPU" if use_gpu else "CPU"
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = thr
    cy.adaptive_min_samples = min(16, samples)
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    cy.denoising_prefilter = "ACCURATE"
    try:
        cy.denoising_quality = "HIGH"
    except Exception:
        pass
    cy.use_light_tree = True

    if fast:
        cy.max_bounces = 6
        cy.diffuse_bounces = 2
        cy.glossy_bounces = 3
        cy.transmission_bounces = 5
        cy.transparent_max_bounces = 8
        cy.volume_bounces = 0
        cy.use_fast_gi = True
        cy.fast_gi_method = "REPLACE"
        cy.ao_bounces_render = 1
    else:
        cy.max_bounces = 10
        cy.diffuse_bounces = 4
        cy.glossy_bounces = 4
        cy.transmission_bounces = 8
        cy.transparent_max_bounces = 12
        cy.volume_bounces = 0
        cy.use_fast_gi = False

    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.blur_glossy = 0.45
    cy.sample_clamp_direct = 0.0
    cy.sample_clamp_indirect = 8.0

    vs = sc.view_settings
    vs.view_transform = "AgX"
    for lk in ("AgX - Medium High Contrast", "AgX - Punchy", "AgX - Base Contrast", "None"):
        try:
            vs.look = lk
            break
        except Exception:
            continue
    vs.exposure = 0.10
    vs.gamma = 1.0

    if frame_end is not None:
        sc.frame_start = 1
        sc.frame_end = frame_end

    if use_gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                if any(d.type == backend for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = True
                    break
            except Exception:
                continue

    setup_cinema_compositor()


def setup_cinema_compositor():
    """Real optical cinema chain: subtle Fog Glow bloom, lens distortion + CA, natural vignette."""
    sc = bpy.context.scene
    try:
        old = bpy.data.node_groups.get("KAIRO_CinemaComp")
        if old is not None:
            bpy.data.node_groups.remove(old)
        ng = bpy.data.node_groups.new("KAIRO_CinemaComp", "CompositorNodeTree")
        rl = ng.nodes.new("CompositorNodeRLayers")
        rl.location = (-600, 0)

        # 1. Subtle optical Fog Glow bloom on sunlit specular highlights & visor LEDs
        glare = ng.nodes.new("CompositorNodeGlare")
        glare.location = (-320, 0)
        for k, v in (("Type", "Fog Glow"), ("Threshold", 1.35), ("Strength", 0.14), ("Size", 0.55), ("Quality", "High")):
            try:
                glare.inputs[k].default_value = v
            except Exception:
                pass

        # 2. Real cinema prime lens micro-distortion + subtle lateral chromatic aberration
        lens = ng.nodes.new("CompositorNodeLensdist")
        lens.location = (-40, 0)
        for k, v in (("Distortion", -0.0035), ("Dispersion", 0.0045)):
            try:
                lens.inputs[k].default_value = v
            except Exception:
                pass
        try:
            lens.use_fit = True
        except Exception:
            pass

        # 3. Gentle optical lens vignette (ellipse mask -> smooth blur -> subtle multiply)
        mask = ng.nodes.new("CompositorNodeEllipseMask")
        mask.location = (-320, -280)
        mask.width = 0.88
        mask.height = 0.88
        blur = ng.nodes.new("CompositorNodeBlur")
        blur.location = (-100, -280)
        for k, v in (("Size", (260, 260)),):
            try:
                blur.inputs[k].default_value = v
            except Exception:
                pass
        try:
            blur.size_x = 260
            blur.size_y = 260
        except Exception:
            pass

        # Remap mask [0, 1] -> [0.80, 1.00] so corners gently darken by 20% like a fast prime lens!
        mr = ng.nodes.new("ShaderNodeMapRange")
        mr.location = (100, -280)
        mr.inputs["To Min"].default_value = 0.80
        mr.inputs["To Max"].default_value = 1.00

        mix_vig = ng.nodes.new("ShaderNodeMix")
        mix_vig.data_type = "RGBA"
        mix_vig.blend_type = "MULTIPLY"
        mix_vig.inputs["Factor"].default_value = 1.0
        mix_vig.location = (280, 0)

        out = ng.nodes.new("NodeGroupOutput")
        out.location = (540, 0)
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")

        ng.links.new(rl.outputs["Image"], glare.inputs["Image"])
        ng.links.new(glare.outputs["Image"], lens.inputs["Image"])
        ng.links.new(mask.outputs["Mask"], blur.inputs["Image"])
        ng.links.new(blur.outputs["Image"], mr.inputs["Value"])
        ng.links.new(lens.outputs["Image"], mix_vig.inputs["A"])
        ng.links.new(mr.outputs["Result"], mix_vig.inputs["B"])
        ng.links.new(mix_vig.outputs["Result"], out.inputs[0])

        sc.compositing_node_group = ng
        sc.render.use_compositing = True
    except Exception as e:
        print("Compositor setup warning:", e)
