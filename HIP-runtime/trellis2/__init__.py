import importlib
__all__ = ["models", "modules", "pipelines", "renderers", "representations", "utils"]
def __getattr__(name):
    if name in __all__:
        module = importlib.import_module("." + name, __name__)
        globals()[name] = module
        return module
    raise AttributeError(name)
