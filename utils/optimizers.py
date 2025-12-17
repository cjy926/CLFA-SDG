import inspect

import torch

from utils import get_class_from_subclasses


class CommonOptimizer:
    """ This optimizer is a wrapper of common optimizers in torch.optim."""

    @staticmethod
    def modify_commandline_options(parser, optimizer_name):
        """ Add new dataset-specific options, and rewrite default values for existing options.

        :param parser: the option parser
        :param optimizer_name: the name of the optimizer
        :return parser: the modified parser
        """
        optimizer_cls = get_class_from_subclasses(torch.optim.Optimizer, optimizer_name, allow_case=True,
                                                  allow_underline=False)
        for arg_name, arg_parameter in inspect.signature(optimizer_cls.__init__).parameters.items():
            if arg_name in ['self', 'params']:
                continue
            default_value = arg_parameter.default
            if isinstance(default_value, bool):
                parser.add_argument('--optimizer_' + arg_name, default=default_value,
                                    type=lambda x: (str(x).lower() == 'true'))
            else:
                parser.add_argument('--optimizer_' + arg_name, default=default_value, type=type(default_value))
        return parser

    def __init__(self, opt, params):
        # opt.optimizer does not start with 'optimizer_'
        optimizer_cls = get_class_from_subclasses(torch.optim.Optimizer, opt.optimizer, allow_case=True,
                                                  allow_underline=False)
        self.optimizer = optimizer_cls(params, **{k.replace('optimizer_', ''): v for k, v in vars(opt).items()
                                                  if k.startswith('optimizer_')})

    def __getattr__(self, name):
        return getattr(self.optimizer, name)
