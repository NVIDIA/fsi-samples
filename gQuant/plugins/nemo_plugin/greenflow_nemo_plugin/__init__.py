from .client import validation, display  # noqa: #401
from greenflow.dataframe_flow._node_flow import register_cleanup


def clean_nemo(ui_clean):
    """
    ui_clean is True if the client send
    'clean' command to the greenflow backend
    """
    try:
        import nemo
        nf = nemo.core.NeuralModuleFactory.get_default_factory()
    except ModuleNotFoundError:
        nf = None
    if nf is not None:
        nf.reset_trainer()
        if ui_clean:
            state = nemo.utils.app_state.AppState()
            state._module_registry.clear()
            state.active_graph.modules.clear()


register_cleanup('cleannemo', clean_nemo)
