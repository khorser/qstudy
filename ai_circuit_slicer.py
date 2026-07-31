"""
AICircuitSlicer -- adds AI-assisted narration and resource estimation on top
of CircuitSlicer, without touching qstudy.py.

Two additive features, wired as new tabs/widgets on the existing Tab:
  1. "Resources" tab: per-slice and cumulative gate counts, depth, and
     entangling-gate counts, computed by resource_stats.slice_resource_stats
     over self.qc (the circuit CircuitSlicer already built and simulated).
  2. "AI Explain" button + output: sends only already-computed facts for the
     current slice (from resource_stats + the entanglement/purity numbers
     CircuitSlicer's `update()` already derives) to Claude, and displays a
     grounded natural-language explanation.

Design choice: subclass + override `sim`/`update` and call super() first,
then layer the new behavior on top, rather than editing qstudy.py. This
keeps the working simulation/instrumentation path untouched and the diff
reviewable in isolation.
"""
import ipywidgets as widgets
from IPython.display import clear_output, HTML
from qiskit.quantum_info import partial_trace, entropy

from qstudy import CircuitSlicer
from resource_stats import slice_resource_stats, render_resource_html
from ai_narrator import SliceFacts, build_slice_prompt, explain_slice


class AICircuitSlicer(CircuitSlicer):
    _ANTHROPIC_MODELS = [
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-fable-5",
        "claude-haiku-4-5-20251001",
    ]

    def __init__(self, algo, *args, anthropic_client=None, model=None,
                 algo_description="", **kwargs):
        """
        anthropic_client: if given, pins the backend to this exact client
        object (skips the in-widget backend selector's client-building --
        useful for scripts/tests). Leave None to get a live "Backend" /
        "Model" / per-backend connection-field selector in the widget
        instead (Anthropic, Ollama, or an OpenAI-compatible aggregator).
        The Anthropic backend's URL/Key fields are optional -- leave blank
        for the real Anthropic API, or point them at any provider that
        speaks Anthropic's native Messages API shape (e.g. OpenRouter,
        confirmed working against https://openrouter.ai/api).
        """
        self._explicit_client = anthropic_client
        self.algo_description = algo_description or getattr(algo, "__doc__", "") or ""

        self.resources = widgets.Output()
        self.ai_button = widgets.Button(description="AI Explain this slice")
        self.ai_output = widgets.Output()
        self.ai_button.on_click(self._on_ai_explain)

        self.backend_dropdown = widgets.Dropdown(
            options=["Anthropic", "Ollama", "OpenAI-compatible"],
            value="Anthropic", description="Backend:"
        )
        self.anthropic_model_combo = widgets.Combobox(
            options=self._ANTHROPIC_MODELS,
            value=model if model in self._ANTHROPIC_MODELS else self._ANTHROPIC_MODELS[0],
            description="Model:", ensure_option=False,
        )
        self.anthropic_refresh_button = widgets.Button(description="Refresh models")
        self.anthropic_status = widgets.HTML(value="")
        self.anthropic_url_text = widgets.Text(
            value="", placeholder="(uses ANTHROPIC_BASE_URL if blank)",
            description="URL:",
        )
        self.anthropic_key_text = widgets.Password(
            value="", placeholder="(uses ANTHROPIC_API_KEY if blank)",
            description="Key:",
        )
        self.ollama_model_dropdown = widgets.Dropdown(
            options=["(click Refresh)"], value="(click Refresh)", description="Model:",
            layout=widgets.Layout(display="none"),
        )
        self.ollama_url_text = widgets.Text(
            value="", placeholder="(uses OLLAMA_API_BASE, else localhost:11434)",
            description="URL:", layout=widgets.Layout(display="none"),
        )
        self.ollama_refresh_button = widgets.Button(
            description="Refresh models", layout=widgets.Layout(display="none"),
        )
        self.ollama_status = widgets.HTML(value="", layout=widgets.Layout(display="none"))
        self.aggregator_model_combo = widgets.Combobox(
            options=[], value="", description="Model:", ensure_option=False,
            layout=widgets.Layout(display="none"),
        )
        self.aggregator_url_text = widgets.Text(
            value="", placeholder="(uses OPENAI_BASE_URL if blank)",
            description="URL:", layout=widgets.Layout(display="none"),
        )
        self.aggregator_key_text = widgets.Password(
            value="", placeholder="(uses OPENAI_API_KEY if blank)",
            description="Key:", layout=widgets.Layout(display="none"),
        )
        self.aggregator_refresh_button = widgets.Button(
            description="Refresh models", layout=widgets.Layout(display="none"),
        )
        self.aggregator_status = widgets.HTML(value="", layout=widgets.Layout(display="none"))
        self.backend_dropdown.observe(self._on_backend_change, names="value")
        self.anthropic_refresh_button.on_click(self._on_anthropic_refresh)
        self.ollama_refresh_button.on_click(self._on_ollama_refresh)
        self.aggregator_refresh_button.on_click(self._on_aggregator_refresh)

        super().__init__(algo, *args, **kwargs)

        # Wire the new tab/controls in after super().__init__ has built self.tab/self.out
        self.tab.children += (self.resources,)
        self.tab.set_title(len(self.tab.children) - 1, "Resources")
        ai_controls = widgets.VBox([
            widgets.HBox([self.backend_dropdown,
                          self.anthropic_url_text, self.anthropic_key_text,
                          self.anthropic_model_combo, self.anthropic_refresh_button,
                          self.ollama_url_text, self.ollama_model_dropdown,
                          self.ollama_refresh_button,
                          self.aggregator_url_text, self.aggregator_key_text,
                          self.aggregator_model_combo, self.aggregator_refresh_button]),
            self.anthropic_status,
            self.ollama_status,
            self.aggregator_status,
        ])
        self.out.children += (ai_controls, widgets.HBox([self.ai_button]), self.ai_output)

        self._refresh_resources()

    def _on_backend_change(self, change):
        backend = change["new"]
        is_anthropic = backend == "Anthropic"
        is_ollama = backend == "Ollama"
        is_aggregator = backend == "OpenAI-compatible"
        # Clear any stale status message left over from a previous backend
        # (e.g. an error from a prior Refresh click) so switching backends
        # doesn't show leftover text from a backend that's no longer selected.
        for status in (self.anthropic_status, self.ollama_status, self.aggregator_status):
            status.value = ""
            status.layout.display = "none"
        for w in (self.anthropic_url_text, self.anthropic_key_text,
                  self.anthropic_model_combo, self.anthropic_refresh_button):
            w.layout.display = "" if is_anthropic else "none"
        for w in (self.ollama_model_dropdown, self.ollama_url_text, self.ollama_refresh_button):
            w.layout.display = "" if is_ollama else "none"
        for w in (self.aggregator_model_combo, self.aggregator_url_text,
                  self.aggregator_key_text, self.aggregator_refresh_button):
            w.layout.display = "" if is_aggregator else "none"
        if is_ollama and self.ollama_model_dropdown.value == "(click Refresh)":
            self._on_ollama_refresh(None)
        if is_aggregator and not self.aggregator_model_combo.options:
            self._on_aggregator_refresh(None)

    def _on_anthropic_refresh(self, _b):
        client, err = self._build_client()
        if err is not None:
            self.anthropic_status.layout.display = ""
            self.anthropic_status.value = f"<i>{err}</i>"
            return
        try:
            models = sorted(m.id for m in client.models.list())
        except Exception as e:
            self.anthropic_status.layout.display = ""
            self.anthropic_status.value = f"<i>Could not list Anthropic models: {e}</i>"
            return
        if not models:
            self.anthropic_status.layout.display = ""
            self.anthropic_status.value = "<i>Anthropic API reachable but returned no models</i>"
            return
        self.anthropic_status.layout.display = "none"
        self.anthropic_status.value = ""
        self.anthropic_model_combo.options = models
        if self.anthropic_model_combo.value not in models:
            self.anthropic_model_combo.value = models[0]

    def _on_ollama_refresh(self, _b):
        from ollama_client import list_ollama_models, resolve_base_url
        resolved_url = resolve_base_url(self.ollama_url_text.value or None)
        try:
            models = list_ollama_models(base_url=resolved_url)
        except Exception as e:
            self.ollama_status.layout.display = ""
            self.ollama_status.value = f"<i>Could not reach Ollama at {resolved_url}: {e}</i>"
            return
        if not models:
            self.ollama_status.layout.display = ""
            self.ollama_status.value = "<i>Ollama reachable but no models installed (ollama pull ...)</i>"
            return
        self.ollama_status.layout.display = "none"
        self.ollama_model_dropdown.options = models
        self.ollama_model_dropdown.value = models[0]

    def _on_aggregator_refresh(self, _b):
        from aggregator_client import list_aggregator_models
        try:
            models = list_aggregator_models(
                base_url=self.aggregator_url_text.value or None,
                api_key=self.aggregator_key_text.value or None,
            )
        except Exception as e:
            self.aggregator_status.layout.display = ""
            self.aggregator_status.value = f"<i>Could not list aggregator models: {e}</i>"
            return
        if not models:
            self.aggregator_status.layout.display = ""
            self.aggregator_status.value = "<i>Aggregator reachable but returned no models</i>"
            return
        self.aggregator_status.layout.display = "none"
        self.aggregator_model_combo.options = models
        if self.aggregator_model_combo.value not in models:
            self.aggregator_model_combo.value = models[0]

    @property
    def _current_model(self):
        backend = self.backend_dropdown.value
        if backend == "Ollama":
            return self.ollama_model_dropdown.value
        if backend == "OpenAI-compatible":
            return self.aggregator_model_combo.value
        return self.anthropic_model_combo.value

    def _build_client(self):
        """Returns (client, error_message). error_message is None on success."""
        if self._explicit_client is not None:
            return self._explicit_client, None

        backend = self.backend_dropdown.value
        if backend == "Anthropic":
            try:
                import anthropic
            except ImportError:
                return None, "anthropic package not installed (pip install anthropic)"
            try:
                return anthropic.Anthropic(
                    base_url=self.anthropic_url_text.value or None,
                    api_key=self.anthropic_key_text.value or None,
                ), None
            except Exception as e:
                return None, f"Could not create Anthropic client (check ANTHROPIC_API_KEY): {e}"
        elif backend == "OpenAI-compatible":
            from aggregator_client import AggregatorClient
            try:
                return AggregatorClient(
                    base_url=self.aggregator_url_text.value or None,
                    api_key=self.aggregator_key_text.value or None,
                ), None
            except Exception as e:
                return None, f"Could not create aggregator client: {e}"
        else:
            from ollama_client import OllamaClient
            return OllamaClient(base_url=self.ollama_url_text.value or None), None

    # -- Resources tab ----------------------------------------------------
    def _refresh_resources(self):
        with self.resources:
            clear_output(wait=True)
            stats = slice_resource_stats(self.qc)
            self._resource_stats = stats  # cached for the AI Explain button
            display(HTML(render_resource_html(stats)))

    def sim(self, change=None):
        super().sim(change)
        # self.qc only exists once the base class has successfully built it
        if getattr(self, "qc", None) is not None:
            self._refresh_resources()

    def update(self, change=None):
        super().update(change)
        # gate counts don't change across slices (same self.qc throughout),
        # but keep this here in case future subclasses want per-slice-switch
        # side effects (e.g. re-highlighting the active row).

    # -- AI Explain ---------------------------------------------------------
    def _current_slice_facts(self):
        label = self.step.value
        stats = getattr(self, "_resource_stats", None) or slice_resource_stats(self.qc)
        slice_stats = stats[label]["slice"]
        cumulative_stats = stats[label]["cumulative"]

        dm = self.res.data()[label + ":dm"]
        entangled = [
            i for i in range(dm.num_qubits)
            if entropy(partial_trace(dm, [j for j in range(dm.num_qubits) if j != i])) > 1e-10
        ]
        pairwise = {}
        for i in range(dm.num_qubits):
            for j in range(i + 1, dm.num_qubits):
                trace_out = [k for k in range(dm.num_qubits) if k not in (i, j)]
                e = entropy(partial_trace(dm, trace_out), base=2)
                if e > 1e-10:
                    pairwise[(i, j)] = e

        return SliceFacts(
            algo_name=type(self.algo).__name__,
            algo_description=self.algo_description,
            label=label,
            gate_counts=slice_stats["gate_counts"],
            entangling_gates=slice_stats["entangling_gates"],
            depth=slice_stats["depth"],
            measurements=slice_stats["measurements"],
            purity=dm.purity().real,
            entangled_qubits=entangled,
            pairwise_entanglement=pairwise,
            cumulative_total_gates=cumulative_stats["total_gates"],
        )

    def _on_ai_explain(self, _b):
        with self.ai_output:
            clear_output(wait=True)
            if self.res is None:
                print("Run a simulation first.")
                return
            facts = self._current_slice_facts()
            client, err = self._build_client()
            if err is not None:
                print(f"{err}\n\nShowing the prompt that would be sent instead:\n")
                print(build_slice_prompt(facts))
                return
            try:
                narration = explain_slice(client, facts, model=self._current_model)
                display(HTML(f"<b>Slice \"{facts.label}\" ({self.backend_dropdown.value}, "
                              f"{self._current_model}):</b><br>{narration}"))
            except Exception as e:
                print(f"AI Explain failed: {e}")
