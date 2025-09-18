from __future__ import annotations
from typing import Any, Dict, List, Optional
import importlib
from langchain_core.runnables.base import RunnableSerializable
from .schemas import TrustInput, TrustOutput, ChatTurn, ShimVariable, ShimQuery

# ---- Lightweight shims to match TRUST's expected object attributes ----

class _ShimQueryObj:
    def __init__(self, question: str):
        self.question = question

class _ShimVariableObj:
    def __init__(self, vid: Optional[str], metadata: Optional[str], queries: Optional[List[_ShimQueryObj]]):
        self.vid = vid
        self.metadata = metadata
        self.queries = queries or []

# ---- Adapter ----

class TrustRunnable(RunnableSerializable[TrustInput, TrustOutput]):
    """
    Wraps TRUST's Conversation agent as a LangChain Runnable.

    Initialization comes from agent.conversation.Conversation(model_name, template_file, simulate_user=None)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        template_file: Optional[str] = None,
        simulate_user: Optional[str] = None,
    ):
        self._default_model_name = model_name
        self._default_template_file = template_file
        self._default_sim_user = simulate_user
        self._conv_cls = None
        self._conv = None
        self._load_trust_class()

        # Defer creating an instance until first call, so we can allow per-request overrides.

    def _load_trust_class(self) -> None:
        import os, sys, importlib

        # 1) Try the modern package path first: trust.agent.conversation
        try:
            mod = importlib.import_module("trust.agent.conversation")
        except ModuleNotFoundError:
            # 2) Fallback: add TRUST/trust to sys.path so "agent.*" works
            here = os.path.abspath(os.path.dirname(__file__))  # .../langserve_trust/trust_adapter
            repo_root = os.path.abspath(os.path.join(here, "..", ".."))  # .../TRUST
            trust_pkg_dir = os.path.join(repo_root, "trust")             # .../TRUST/trust
            if trust_pkg_dir not in sys.path:
                sys.path.insert(0, trust_pkg_dir)
            # Now import using the flat 'agent' path that conversation.py expects
            mod = importlib.import_module("agent.conversation")

        conv_cls = getattr(mod, "Conversation", None)
        if conv_cls is None:
            raise RuntimeError("Conversation class not found (looked in trust.agent.conversation / agent.conversation).")
        self._conv_cls = conv_cls


    def _ensure_conversation(self, model_name: Optional[str], template_file: Optional[str], simulate_user: Optional[str]):
        # Use defaults unless overrides provided in the request
        model = model_name or self._default_model_name
        tmpl = template_file or self._default_template_file
        sim  = simulate_user if simulate_user is not None else self._default_sim_user

        if not model:
            raise ValueError("Conversation requires 'model_name'. Provide default in server.py or pass in request.")
        if not tmpl:
            raise ValueError("Conversation requires 'template_file'. Provide default in server.py or pass in request.")

        # Recreate if missing or model/template overrides differ
        if (
            self._conv is None
            or getattr(self._conv, "model_name", None) != model
            or getattr(self._conv, "conv_templates", None) is None  # if template not loaded
        ):
            self._conv = self._conv_cls(model, tmpl, sim)

    def _as_history(self, hist: Optional[List[ChatTurn]]) -> List[str]:
        """
        Conversation.format_prompt() examples show {history} being treated as a textual block.
        Upstream code often passes structured history; here we flatten to a simple list
        like: ["user: ...", "assistant: ..."] which Conversation's prompt formatters
        turn into the expected strings.
        """
        if not hist:
            return []
        return [f"{t.role}: {t.content}" for t in hist]

    def _var_from_schema(self, v: Optional[ShimVariable]) -> Optional[_ShimVariableObj]:
        if v is None:
            return None
        queries = [_ShimQueryObj(q.question) for q in (v.queries or [])]
        return _ShimVariableObj(v.vid, v.metadata, queries)

    def invoke(self, input: TrustInput, config: Optional[Dict[str, Any]] = None) -> TrustOutput:
        # Normalize empty strings from schema into None so defaults kick in
        model_name    = input.model_name or None
        template_file = input.template_file or None
        simulate_user = input.simulate_user or None

        # Allow per-request overrides of Conversation init params
        self._ensure_conversation(
            model_name=model_name,
            template_file=template_file,
            simulate_user=simulate_user,
        )

        op = input.op
        history = self._as_history(input.history)

        if op == "da_tags":
            if not input.questions:
                raise ValueError("op=da_tags requires 'questions': List[str]")

            # Normalize to objects that have `.question` to match TRUST expectations
            qobjs = [_ShimQueryObj(q) for q in input.questions]  # strings -> obj
            for q in input.questions:
                if isinstance(q, str):
                    qobjs.append(_ShimQueryObj(q))
                elif isinstance(q, ShimQuery):
                    qobjs.append(_ShimQueryObj(q.question))
                else:
                    qobjs.append(_ShimQueryObj(str(q)))

            tags = self._conv.da_tags(history=history, questions=qobjs)
            return TrustOutput(text=",".join(tags), state={"tags": tags})

        elif op == "conv_transition":
            var_obj = self._var_from_schema(input.variable)
            if var_obj is None:
                raise ValueError("op=conv_transition requires 'variable'")
            text = self._conv.conv_transition(history=history, variable=var_obj)
            return TrustOutput(text=text, state={})

        elif op == "conv_question":
            if input.cur_query is None:
                raise ValueError("op=conv_question requires 'cur_query'")
            if input.cur_tags is None:
                raise ValueError("op=conv_question requires 'cur_tags'")
            var_obj = self._var_from_schema(input.variable)
            if var_obj is None:
                raise ValueError("op=conv_question requires 'variable'")
            qobj = _ShimQueryObj(input.cur_query.question)
            text = self._conv.conv_question(
                cur_query=qobj,
                cur_tags=input.cur_tags,
                history=history,
                variable=var_obj,
            )
            return TrustOutput(text=text, state={})

        else:
            raise ValueError(
                "Unsupported op. Use one of: 'da_tags', 'conv_transition', 'conv_question'."
            )
