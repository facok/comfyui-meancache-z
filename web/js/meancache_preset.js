import { app } from "../../../scripts/app.js";

// Preset parameter values (must mirror Python PRESETS in meancache_node.py)
// Tuned to match official MeanCache acceleration ratios.
const PRESETS = {
    "Quality":  { rel_l1_thresh: 0.15, skip_budget: 0.20, start_step: 3, end_step: -1, enable_pssp: true, peak_threshold: 0.08, adaptive_k: true },
    "Balanced": { rel_l1_thresh: 0.30, skip_budget: 0.40, start_step: 2, end_step: -1, enable_pssp: true, peak_threshold: 0.15, adaptive_k: true },
    "Speed":    { rel_l1_thresh: 0.50, skip_budget: 0.55, start_step: 1, end_step: -1, enable_pssp: true, peak_threshold: 0.35, adaptive_k: true },
    "Turbo":    { rel_l1_thresh: 0.55, skip_budget: 0.60, start_step: 1, end_step: -1, enable_pssp: true, peak_threshold: 0.45, adaptive_k: true },
};

// Widget names controlled by preset selection
const TUNING_WIDGETS = ["rel_l1_thresh", "skip_budget", "start_step", "end_step", "enable_pssp", "peak_threshold", "adaptive_k"];

function toggleWidget(widget, show) {
    if (!widget) return;
    // Store original properties on first hide
    if (!widget._mc_origProps) {
        widget._mc_origProps = {
            type: widget.type,
            computeSize: widget.computeSize,
        };
    }
    if (show) {
        widget.type = widget._mc_origProps.type;
        widget.computeSize = widget._mc_origProps.computeSize;
    } else {
        widget.type = "mc_hidden";
        widget.computeSize = () => [0, -4];
    }
}

function syncPreset(node, presetName) {
    const isCustom = !(presetName in PRESETS);
    const preset = PRESETS[presetName];

    for (const name of TUNING_WIDGETS) {
        const widget = node.widgets?.find(w => w.name === name);
        if (!widget) continue;

        // Sync values from preset
        if (preset && name in preset) {
            widget.value = preset[name];
        }

        // Show only in Custom mode
        toggleWidget(widget, isCustom);
    }

    // Resize node to fit visible widgets
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
        node.graph?.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "MeanCache.PresetSync",

    nodeCreated(node) {
        if (node.comfyClass !== "MeanCache_ZImage") return;

        const presetWidget = node.widgets?.find(w => w.name === "preset");
        if (!presetWidget) return;

        // Intercept value changes via property descriptor
        let currentValue = presetWidget.value;
        Object.defineProperty(presetWidget, "value", {
            get() {
                return currentValue;
            },
            set(newVal) {
                currentValue = newVal;
                syncPreset(node, newVal);
            },
            configurable: true,
        });

        // Initial sync (use setTimeout to ensure widgets are fully initialized)
        setTimeout(() => syncPreset(node, currentValue), 0);
    },
});
