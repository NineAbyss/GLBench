from string import Formatter

from omegaconf import OmegaConf, DictConfig
from pandas import DataFrame


def get_string_args(s):
    return [fn for _, fn, _, _ in Formatter().parse(s) if fn is not None]


def preprocess_prompt_config(prompt_cfg, lookup_dict):
    cfg_dict = OmegaConf.to_object(prompt_cfg)
    processed_dict = {}
    for k, v in cfg_dict.items():
        if not k.startswith('_') and k in lookup_dict:  # Is a prompt template
            retrieved_str = lookup_dict[k][v]
            processed_dict[k] = preprocess_yaml_fstring(retrieved_str)
    return processed_dict


def preprocess_yaml_fstring(s):
    s = s.replace('\n', '')  # Remove original \n
    s = s.replace('\\n ', '\n')  # Note that there is a SPACE after \\n
    s = s.replace('\\n', '\n')
    return s


def init_prompt_from_cfg(prompt_cfg: DictConfig, template_lookup_dict, **kwargs):
    cfg_dict = preprocess_prompt_config(prompt_cfg, template_lookup_dict)
    template = cfg_dict.pop('prompt')
    cfg_dict.update({k: v for k, v in kwargs.items() if not k.startswith('_')})
    return Prompt(template, **cfg_dict)


class Prompt:
    # A simpler class than Langchain.PromptTemplate
    # With some tweaks
    def __init__(self, template: str, **prompt_args):
        self.template = preprocess_yaml_fstring(template)
        self._variables = {k: preprocess_yaml_fstring(v) for k, v in prompt_args.items()}
        self._var_set = set(get_string_args(self.template))

    def update(self, **kwargs):
        self._variables.update(kwargs)

    @property
    def filled_template(self, **kwargs):
        return self.__call__(**kwargs, assert_vars=False)

    @property
    def unfilled_fields(self):
        return list(set(self._variables.keys()) - self._var_set)

    def __call__(self, assert_vars=True, **kwargs):
        args = {**self._variables, **kwargs}
        if assert_vars:
            assert len(set(args.keys()) - self._var_set) >= 0, f'{self._var_set - set(args.keys())} not given.'
        else:  # For unknown args, keep the {arg} instead.
            args = {k: args.get(k, f'{{{k}}}') for k in self._var_set}
        return self.template.format(**args)

    def __str__(self):
        return self.filled_template

    def __repr__(self):
        return f'PromptTemplate: <<{self.__str__()}>>'

