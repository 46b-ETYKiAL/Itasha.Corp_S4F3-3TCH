"""Widget layout optimizer for ComfyUI custom nodes.

Groups related inputs, assigns optimal widget types based on
parameter semantics, applies progressive disclosure, and generates
Vue component stubs for frontend widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import InputSpec, NodeSpec, WidgetType

# Semantic groups: parameter name keywords → group label
_SEMANTIC_GROUPS: dict[str, list[str]] = {
    "dimensions": ["width", "height", "size", "resolution", "scale"],
    "color": ["color", "hue", "saturation", "brightness", "red", "green", "blue", "alpha", "rgb", "hsv"],
    "position": ["x", "y", "z", "left", "top", "right", "bottom", "offset"],
    "transform": ["rotation", "angle", "rotate", "flip", "mirror", "skew"],
    "filter": ["blur", "sharpen", "denoise", "smooth", "radius", "sigma", "kernel"],
    "blend": ["opacity", "blend", "mix", "weight", "strength", "amount", "intensity"],
    "text": ["text", "prompt", "label", "name", "title", "caption", "description"],
    "model": ["model", "checkpoint", "lora", "vae", "clip"],
}

# Keyword → optimal widget type mapping
_SEMANTIC_WIDGET_MAP: dict[str, WidgetType] = {
    "enabled": WidgetType.BOOLEAN,
    "toggle": WidgetType.BOOLEAN,
    "active": WidgetType.BOOLEAN,
    "visible": WidgetType.BOOLEAN,
    "invert": WidgetType.BOOLEAN,
    "flip": WidgetType.BOOLEAN,
    "mode": WidgetType.COMBO,
    "method": WidgetType.COMBO,
    "type": WidgetType.COMBO,
    "style": WidgetType.COMBO,
    "algorithm": WidgetType.COMBO,
    "interpolation": WidgetType.COMBO,
    "seed": WidgetType.INT,
    "steps": WidgetType.INT,
    "count": WidgetType.INT,
    "width": WidgetType.INT,
    "height": WidgetType.INT,
    "batch_size": WidgetType.INT,
    "opacity": WidgetType.FLOAT,
    "strength": WidgetType.FLOAT,
    "weight": WidgetType.FLOAT,
    "cfg": WidgetType.FLOAT,
    "denoise": WidgetType.FLOAT,
    "prompt": WidgetType.STRING,
    "text": WidgetType.STRING,
    "name": WidgetType.STRING,
}

# Progressive disclosure threshold
_PROGRESSIVE_DISCLOSURE_THRESHOLD = 8


@dataclass
class InputGroup:
    """A logical grouping of related inputs.

    Attributes:
        label: Human-readable group name.
        inputs: Ordered list of inputs in this group.
    """

    label: str
    inputs: list[InputSpec] = field(default_factory=list)


@dataclass
class LayoutResult:
    """Result from layout optimization.

    Attributes:
        groups: Ordered list of input groups.
        hidden_inputs: Inputs marked as hidden for progressive disclosure.
        vue_stub: Generated Vue component stub string.
    """

    groups: list[InputGroup] = field(default_factory=list)
    hidden_inputs: list[str] = field(default_factory=list)
    vue_stub: str = ""


def _find_group(name: str) -> str:
    """Determine which semantic group an input belongs to.

    Args:
        name: Input parameter name (snake_case).

    Returns:
        Group label or "other" if no match.
    """
    name_lower = name.lower()
    for group_label, keywords in _SEMANTIC_GROUPS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return group_label
    return "other"


def _optimize_widget_type(inp: InputSpec) -> InputSpec:
    """Assign optimal widget type based on parameter semantics.

    Args:
        inp: Original input specification.

    Returns:
        New InputSpec with potentially updated widget type.
    """
    name_lower = inp.name.lower()
    for keyword, widget_type in _SEMANTIC_WIDGET_MAP.items():
        if keyword in name_lower:
            # Only override if the current type is generic
            if inp.widget.widget_type in (WidgetType.FLOAT, WidgetType.STRING):
                new_widget = inp.widget.model_copy(update={"widget_type": widget_type})
                return inp.model_copy(update={"widget": new_widget})
            break
    return inp


def group_inputs(inputs: list[InputSpec]) -> list[InputGroup]:
    """Group related inputs together by semantic category.

    Args:
        inputs: Flat list of input specifications.

    Returns:
        Ordered list of InputGroup objects.
    """
    groups: dict[str, InputGroup] = {}
    for inp in inputs:
        group_label = _find_group(inp.name)
        if group_label not in groups:
            groups[group_label] = InputGroup(label=group_label)
        groups[group_label].inputs.append(inp)

    # Sort: known groups first, then "other" last
    known_order = list(_SEMANTIC_GROUPS.keys())
    result: list[InputGroup] = []
    for label in known_order:
        if label in groups:
            result.append(groups[label])
    if "other" in groups:
        result.append(groups["other"])

    return result


def apply_progressive_disclosure(
    inputs: list[InputSpec],
    threshold: int = _PROGRESSIVE_DISCLOSURE_THRESHOLD,
) -> list[InputSpec]:
    """Mark excess inputs as hidden for progressive disclosure.

    Required inputs and the first `threshold` inputs remain visible.
    Inputs beyond the threshold are marked hidden.

    Args:
        inputs: List of input specifications.
        threshold: Number of visible inputs before hiding.

    Returns:
        Updated list with hidden flags set.
    """
    visible_count = 0
    result: list[InputSpec] = []

    for inp in inputs:
        if inp.required or visible_count < threshold:
            result.append(inp)
            visible_count += 1
        else:
            new_widget = inp.widget.model_copy(update={"hidden": True})
            result.append(inp.model_copy(update={"widget": new_widget}))

    return result


def optimize_layout(spec: NodeSpec) -> LayoutResult:
    """Run full layout optimization on a node specification.

    Applies semantic widget type assignment, input grouping, and
    progressive disclosure.

    Args:
        spec: Complete node specification.

    Returns:
        LayoutResult with groups, hidden inputs, and Vue stub.
    """
    # Optimize widget types
    optimized_inputs = [_optimize_widget_type(inp) for inp in spec.inputs]

    # Apply progressive disclosure
    disclosed_inputs = apply_progressive_disclosure(optimized_inputs)

    # Group inputs
    groups = group_inputs(disclosed_inputs)

    # Collect hidden input names
    hidden_names = [inp.name for inp in disclosed_inputs if inp.widget.hidden]

    # Generate Vue stub
    vue_stub = generate_vue_stub(spec.name, groups)

    return LayoutResult(
        groups=groups,
        hidden_inputs=hidden_names,
        vue_stub=vue_stub,
    )


def generate_vue_stub(node_name: str, groups: list[InputGroup]) -> str:
    """Generate a Vue 3 SFC component stub for the node widget.

    Args:
        node_name: The node's internal name.
        groups: Grouped inputs from layout optimization.

    Returns:
        Vue SFC template string.
    """
    sections: list[str] = []
    for group in groups:
        inputs_html: list[str] = []
        for inp in group.inputs:
            widget = inp.widget
            if widget.hidden:
                inputs_html.append('      <div v-show="showAdvanced" class="input-row">')
            else:
                inputs_html.append('      <div class="input-row">')
            inputs_html.append(f"        <label>{inp.name}</label>")
            # Widget component based on type
            if widget.widget_type == WidgetType.BOOLEAN:
                inputs_html.append(f'        <toggle-input v-model="values.{inp.name}" />')
            elif widget.widget_type == WidgetType.COMBO:
                inputs_html.append(f'        <combo-input v-model="values.{inp.name}" :options="options.{inp.name}" />')
            elif widget.widget_type in (WidgetType.INT, WidgetType.FLOAT):
                inputs_html.append(
                    f'        <slider-input v-model="values.{inp.name}" '
                    f':min="{widget.min_value or 0}" '
                    f':max="{widget.max_value or 100}" '
                    f':step="{widget.step or 1}" />'
                )
            elif widget.widget_type == WidgetType.STRING:
                tag = "textarea-input" if widget.multiline else "text-input"
                inputs_html.append(f'        <{tag} v-model="values.{inp.name}" />')
            else:
                inputs_html.append(f"        <!-- slot: {inp.widget.widget_type.value} -->")
            inputs_html.append("      </div>")

        section = (
            f'    <fieldset class="input-group">\n'
            f"      <legend>{group.label.title()}</legend>\n" + "\n".join(inputs_html) + "\n    </fieldset>"
        )
        sections.append(section)

    template_body = "\n".join(sections)

    return f"""\
<template>
  <div class="node-widget {node_name}">
{template_body}
    <button v-if="hasHidden" @click="showAdvanced = !showAdvanced">
      {{{{ showAdvanced ? 'Hide Advanced' : 'Show Advanced' }}}}
    </button>
  </div>
</template>

<script setup lang="ts">
import {{ ref, reactive }} from 'vue';

const props = defineProps<{{
  nodeId: string;
}}>()

const showAdvanced = ref(false);
const hasHidden = {str(any(inp.widget.hidden for g in groups for inp in g.inputs)).lower()};
const values = reactive({{}});
</script>

<style scoped>
.node-widget {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.input-group {{
  border: 1px solid var(--border-color, #333);
  border-radius: 4px;
  padding: 8px;
}}
.input-row {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 0;
}}
</style>
"""
