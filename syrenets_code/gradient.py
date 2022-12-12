import torch
import torch.autograd.functional as fct


def _check_requires_grad(inputs, input_type, strict):
    # Used to make all the necessary checks to raise nice errors in strict mode.
    if not strict:
        return

    if input_type not in ["tensors"]:
        raise RuntimeError("Invalid input_type to _check_requires_grad")
    for i, inp in enumerate(inputs):
        if inp is None:
            # This can only be reached for grad_inputs.
            raise RuntimeError("The output of the user-provided function is independent of input {}."
                               " This is not allowed in strict mode.".format(i))
        if not inp.requires_grad:
            if input_type == "tensors":
                raise RuntimeError("The user-provided tensor is independent of the input {}. This is not allowed in strict mode.".format(i))
            else:
                raise RuntimeError("{}-th tensor of the user-provided tensors does not require gradients."
                                   " The tensors must be computed in a differentiable manner from the inputs"
                                   " when running in strict mode.".format(i))


def gradient(tensors, inputs, create_graph=False, strict=False):
    r"""Function that computes the Jacobian of functions that produced the given tensors.

    Args:
        tensors (sequence of Tensors or Tensor): the result of functions that take Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs (arguments) wrt. which to differentiate.
        create_graph (bool, optional): If ``True``, the tensor_grad will be
            computed in a differentiable manner. Note that when ``strict`` is
            ``False``, the result can not require gradients or be disconnected
            from the inputs.  Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the tensors are
            independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        tensor gradient (Tensor or nested tuple of Tensors): if there are a single
            input and output, this will be a single Tensor containing the
            gradient for the linearized inputs and output. If one of the two is
            a tuple, then the gradient will be a tuple of Tensors. If both of
            them are tuples, then the Jacobian will be a tuple of tuple of
            Tensors where ``gradient[i][j]`` will contain the gradient of the
            ``i``\th output and ``j``\th input and will have as size the
            concatenation of the sizes of the corresponding output and the
            corresponding input.
    """
    is_args_tuple, inputs = fct._as_tuple(inputs, "inputs", "tensor_grad")
    is_tensors_tuple, tensors = fct._as_tuple(tensors, "tensors", "tensor_grad")
    _check_requires_grad(tensors, "tensors", strict=strict)
    grads = tuple()

    for i, tensor in enumerate(tensors):

        grad = tuple([] for _ in range(len(inputs)))
        for j in range(tensor.nelement()):     # denotes j-th element of the tensor
            grad_j = fct._autograd_grad((tensor.reshape(-1)[j], ), inputs, retain_graph=True, create_graph=create_graph)

            for el_jk, (grad__k, grad_jk, inputs_k) in enumerate(zip(grad, grad_j, inputs)):   # k denotes gradient wrt. k-th argument

                if grad_jk is not None:
                    if strict and create_graph and not grad_jk.requires_grad:
                        msg = ("The grad of the tensor {} of the user-provided tuple of tensors is independent of input {}."
                               " This is not allowed in strict mode when create_graph=True.".format(i, el_jk))
                        raise RuntimeError(msg)
                    grad__k.append(grad_jk)
                else:
                    if strict:
                        msg = ("The tensor {} of the user-provided tuple of tensors is independent of input {}."
                               " This is not allowed in strict mode.".format(i, el_jk))
                        raise RuntimeError(msg)
                    grad__k.append(torch.zeros_like(inputs_k))

        grads += (tuple(torch.stack(grad__k, dim=0).view(tensor.size() + inputs[el_jk].size()) for (el_jk, grad__k) in enumerate(grad)), )

    grads = fct._grad_postprocess(grads, create_graph)

    return fct._tuple_postprocess(grads, (is_tensors_tuple, is_args_tuple))
