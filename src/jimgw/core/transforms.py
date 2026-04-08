from abc import ABC
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.scipy.special import logit
from beartype import beartype as typechecker
from jaxtyping import Float, Array, jaxtyped


class Transform(ABC):
    """Abstract base class for parameter-space transforms.

    Transforms are responsible for mapping named parameter dictionaries between
    two coordinate spaces. Subclasses implement the actual numerical mapping;
    this base class handles name bookkeeping.

    Attributes:
        name_mapping (tuple[list[str], list[str]]): A pair ``(from_names, to_names)``
            describing which input parameters are consumed and which output parameters
            are produced.
    """

    name_mapping: tuple[list[str], list[str]]

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ) -> None:
        """
        Args:
            name_mapping (tuple[list[str], list[str]]): Pair of
                ``(input_parameter_names, output_parameter_names)``.
        """
        self.name_mapping = name_mapping

    def propagate_name(self, x: Sequence[str]) -> tuple[str, ...]:
        # Remove names in from_list, then append names in to_list (preserving order)
        result = [name for name in x if name not in self.name_mapping[0]]
        result += [name for name in self.name_mapping[1] if name not in result]
        return tuple(result)


class NtoMTransform(Transform):
    """N-to-M transform: consumes N named parameters and produces M named parameters.

    Only a forward pass is defined. This is the most general transform type.

    Attributes:
        transform_func (Callable): Callable that maps an input dict to an output dict.
    """

    transform_func: Callable[[dict[str, Float]], dict[str, Float]]

    def __repr__(self):
        return f"NtoMTransform(name_mapping={self.name_mapping})"

    def forward(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Push forward the input x to transformed coordinate y.

        Args:
            x (dict[str, Float]): The input dictionary.

        Returns:
            dict[str, Float]: The transformed dictionary.
        """
        x_copy = x.copy()
        output_params = self.transform_func(x_copy)
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy


class NtoNTransform(NtoMTransform):
    """N-to-N transform with automatic Jacobian computation via forward-mode AD.

    The number of input and output parameters must be equal. The log-Jacobian
    determinant of the transform is computed automatically using ``jax.jacfwd``.
    """

    def __repr__(self):
        return f"NtoNTransform(name_mapping={self.name_mapping})"

    @property
    def n_dim(self) -> int:
        """Number of parameters consumed/produced by this transform."""
        return len(self.name_mapping[0])

    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """
        Transform the input x to transformed coordinate y and return the log Jacobian determinant.
        This only works if the transform is a N -> N transform.

        Args:
            x (dict[str, Float]): The input dictionary.

        Returns:
            tuple[dict[str, Float], Float]: The transformed dictionary and the log Jacobian determinant.
        """
        x_copy = x.copy()
        transform_params = dict((key, x_copy[key]) for key in self.name_mapping[0])
        output_params = self.transform_func(transform_params)
        jacobian = jax.jacfwd(self.transform_func)(transform_params)
        jacobian = jnp.array(jax.tree.leaves(jacobian))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy, jacobian


class BijectiveTransform(NtoNTransform):
    """Bijective (invertible) N-to-N transform.

    Extends :class:`NtoNTransform` with an inverse transform function and
    corresponding ``inverse`` and ``backward`` methods.

    Attributes:
        inverse_transform_func (Callable): Callable implementing the inverse map.
    """

    inverse_transform_func: Callable[[dict[str, Float]], dict[str, Float]]

    def __repr__(self):
        return f"BijectiveTransform(name_mapping={self.name_mapping})"

    def inverse(self, y: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """
        Inverse transform the input y to original coordinate x.

        Args:
            y (dict[str, Float]): The transformed dictionary.

        Returns:
            tuple[dict[str, Float], Float]: The original dictionary and the log Jacobian determinant.
        """
        y_copy = y.copy()
        transform_params = dict((key, y_copy[key]) for key in self.name_mapping[1])
        output_params = self.inverse_transform_func(transform_params)
        jacobian = jax.jacfwd(self.inverse_transform_func)(transform_params)
        jacobian = jnp.array(jax.tree.leaves(jacobian))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy, jacobian

    def backward(self, y: dict[str, Float]) -> dict[str, Float]:
        """
        Pull back the input y to original coordinate x.

        Args:
            y (dict[str, Float]): The transformed dictionary.

        Returns:
            dict[str, Float]: The original dictionary.
        """
        y_copy = y.copy()
        output_params = self.inverse_transform_func(y_copy)
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy


class ConditionalBijectiveTransform(BijectiveTransform):
    """Bijective transform that depends on additional conditioning parameters.

    Like :class:`BijectiveTransform`, but the transform and inverse functions
    also receive a set of fixed conditioning parameters (e.g. masses when
    converting spin angles). The Jacobian is computed only with respect to the
    primary (non-conditioning) parameters.

    Attributes:
        conditional_names (list[str]): Names of the conditioning parameters.
    """

    conditional_names: list[str]

    def __repr__(self):
        return f"ConditionalBijectiveTransform(name_mapping={self.name_mapping}, conditional_names={self.conditional_names})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        conditional_names: list[str],
    ) -> None:
        """
        Args:
            name_mapping (tuple[list[str], list[str]]): Pair of
                ``(input_names, output_names)`` for the primary parameters.
            conditional_names (list[str]): Names of the conditioning parameters
                that are passed through unchanged.
        """
        super().__init__(name_mapping)
        self.conditional_names = conditional_names

    def transform(self, x: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """Apply the conditional forward transform and compute the log-Jacobian.

        Args:
            x (dict[str, Float]): Parameter dictionary containing both primary and
                conditioning parameters.

        Returns:
            tuple[dict[str, Float], Float]: The transformed parameter dictionary and
                the log absolute Jacobian determinant w.r.t. the primary parameters.
        """
        x_copy = x.copy()
        transform_params = dict((key, x_copy[key]) for key in self.name_mapping[0])
        transform_params.update(
            dict((key, x_copy[key]) for key in self.conditional_names)
        )
        output_params = self.transform_func(transform_params)
        jacobian = jax.jacfwd(self.transform_func)(transform_params)
        jacobian_copy = {
            key1: {key2: jacobian[key1][key2] for key2 in self.name_mapping[0]}
            for key1 in self.name_mapping[1]
        }
        jacobian = jnp.array(jax.tree.leaves(jacobian_copy))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: x_copy.pop(key),
            self.name_mapping[0],
        )
        jax.tree.map(
            lambda key: x_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return x_copy, jacobian

    def inverse(self, y: dict[str, Float]) -> tuple[dict[str, Float], Float]:
        """Apply the conditional inverse transform and compute the log-Jacobian.

        Args:
            y (dict[str, Float]): Parameter dictionary in the output space, containing
                both primary (output) and conditioning parameters.

        Returns:
            tuple[dict[str, Float], Float]: The inverse-transformed parameter dictionary
                and the log absolute Jacobian determinant w.r.t. the primary output
                parameters.
        """
        y_copy = y.copy()
        transform_params = dict((key, y_copy[key]) for key in self.name_mapping[1])
        transform_params.update(
            dict((key, y_copy[key]) for key in self.conditional_names)
        )
        output_params = self.inverse_transform_func(transform_params)
        jacobian = jax.jacfwd(self.inverse_transform_func)(transform_params)
        jacobian_copy = {
            key1: {key2: jacobian[key1][key2] for key2 in self.name_mapping[1]}
            for key1 in self.name_mapping[0]
        }
        jacobian = jnp.array(jax.tree.leaves(jacobian_copy))
        jacobian = jnp.log(
            jnp.absolute(jnp.linalg.det(jacobian.reshape(self.n_dim, self.n_dim)))
        )
        jax.tree.map(
            lambda key: y_copy.pop(key),
            self.name_mapping[1],
        )
        jax.tree.map(
            lambda key: y_copy.update({key: output_params[key]}),
            list(output_params.keys()),
        )
        return y_copy, jacobian


@jaxtyped(typechecker=typechecker)
class ScaleTransform(BijectiveTransform):
    """Elementwise scaling transform: ``y = x * scale``.

    Attributes:
        scale (Float): Multiplicative scale factor.
    """

    scale: Float

    def __repr__(self):
        return f"ScaleTransform(name_mapping={self.name_mapping}, scale={self.scale})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        scale: Float,
    ) -> None:
        """
        Args:
            name_mapping (tuple[list[str], list[str]]): ``(input_names, output_names)``.
            scale (Float): Scale factor applied in the forward direction.
        """
        super().__init__(name_mapping)
        self.scale = scale
        self.transform_func = lambda x: {
            name_mapping[1][i]: x[name_mapping[0][i]] * self.scale
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: x[name_mapping[1][i]] / self.scale
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class OffsetTransform(BijectiveTransform):
    """Elementwise offset (translation) transform: ``y = x + offset``.

    Attributes:
        offset (Float): Additive offset applied in the forward direction.
    """

    offset: Float

    def __repr__(self):
        return (
            f"OffsetTransform(name_mapping={self.name_mapping}, offset={self.offset})"
        )

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        offset: Float,
    ) -> None:
        """
        Args:
            name_mapping (tuple[list[str], list[str]]): ``(input_names, output_names)``.
            offset (Float): Offset added in the forward direction.
        """
        super().__init__(name_mapping)
        self.offset = offset
        self.transform_func = lambda x: {
            name_mapping[1][i]: x[name_mapping[0][i]] + self.offset
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: x[name_mapping[1][i]] - self.offset
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class LogitTransform(BijectiveTransform):
    """
    Logit transform.

    Args:
        name_mapping (tuple[list[str], list[str]]): The name mapping between the input and output dictionary.
    """

    def __repr__(self):
        return f"LogitTransform(name_mapping={self.name_mapping})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        self.transform_func = lambda x: {
            name_mapping[1][i]: 1 / (1 + jnp.exp(-x[name_mapping[0][i]]))
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: logit(x[name_mapping[1][i]])
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class SineTransform(BijectiveTransform):
    """
    Sine transformation.

    The original parameter is expected to be in [-pi/2, pi/2].

    Args:
        name_mapping (tuple[list[str], list[str]]): The name mapping between the input and output dictionary.
    """

    def __repr__(self):
        return f"SineTransform(name_mapping={self.name_mapping})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        self.transform_func = lambda x: {
            name_mapping[1][i]: jnp.sin(x[name_mapping[0][i]])
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.arcsin(x[name_mapping[1][i]])
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class CosineTransform(BijectiveTransform):
    """
    Cosine transformation.

    The original parameter is expected to be in [0, pi].

    Args:
        name_mapping (tuple[list[str], list[str]]): The name mapping between the input and output dictionary.
    """

    def __repr__(self):
        return f"CosineTransform(name_mapping={self.name_mapping})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
    ):
        super().__init__(name_mapping)
        self.transform_func = lambda x: {
            name_mapping[1][i]: jnp.cos(x[name_mapping[0][i]])
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.arccos(x[name_mapping[1][i]])
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class BoundToBound(BijectiveTransform):
    """Linear rescaling from one bounded interval to another.

    Maps ``x ∈ [original_lower, original_upper]`` to
    ``y ∈ [target_lower, target_upper]`` via a linear (affine) transform.

    Attributes:
        original_lower_bound (Float[Array, " n_dim"]): Lower bound(s) of the input interval.
        original_upper_bound (Float[Array, " n_dim"]): Upper bound(s) of the input interval.
        target_lower_bound (Float[Array, " n_dim"]): Lower bound(s) of the output interval.
        target_upper_bound (Float[Array, " n_dim"]): Upper bound(s) of the output interval.
    """

    original_lower_bound: Float[Array, " n_dim"]
    original_upper_bound: Float[Array, " n_dim"]
    target_lower_bound: Float[Array, " n_dim"]
    target_upper_bound: Float[Array, " n_dim"]

    def __repr__(self):
        return f"BoundToBound(name_mapping={self.name_mapping}, original_lower_bound={self.original_lower_bound}, original_upper_bound={self.original_upper_bound}, target_lower_bound={self.target_lower_bound}, target_upper_bound={self.target_upper_bound})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        original_lower_bound: Float | Float[Array, " n_dim"],
        original_upper_bound: Float | Float[Array, " n_dim"],
        target_lower_bound: Float | Float[Array, " n_dim"],
        target_upper_bound: Float | Float[Array, " n_dim"],
    ):
        super().__init__(name_mapping)
        self.original_lower_bound = jnp.atleast_1d(original_lower_bound)
        self.original_upper_bound = jnp.atleast_1d(original_upper_bound)
        self.target_lower_bound = jnp.atleast_1d(target_lower_bound)
        self.target_upper_bound = jnp.atleast_1d(target_upper_bound)

        self.transform_func = lambda x: {
            name_mapping[1][i]: (x[name_mapping[0][i]] - self.original_lower_bound[i])
            * (self.target_upper_bound[i] - self.target_lower_bound[i])
            / (self.original_upper_bound[i] - self.original_lower_bound[i])
            + self.target_lower_bound[i]
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: (x[name_mapping[1][i]] - self.target_lower_bound[i])
            * (self.original_upper_bound[i] - self.original_lower_bound[i])
            / (self.target_upper_bound[i] - self.target_lower_bound[i])
            + self.original_lower_bound[i]
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class BoundToUnbound(BijectiveTransform):
    """Logit-based transform from a bounded interval to the real line.

    Maps ``x ∈ (original_lower, original_upper)`` to ``y ∈ (-∞, +∞)`` via

    .. math::

        y = \\text{logit}\\!\\left(\\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}\\right).

    The inverse maps back with the sigmoid function.

    Attributes:
        original_lower_bound (Float[Array, " n_dim"]): Lower bound(s) of the input interval.
        original_upper_bound (Float[Array, " n_dim"]): Upper bound(s) of the input interval.
    """

    original_lower_bound: Float[Array, " n_dim"]
    original_upper_bound: Float[Array, " n_dim"]

    def __repr__(self):
        return f"BoundToUnbound(name_mapping={self.name_mapping}, original_lower_bound={self.original_lower_bound}, original_upper_bound={self.original_upper_bound})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        original_lower_bound: Float | Float[Array, " n_dim"],
        original_upper_bound: Float | Float[Array, " n_dim"],
    ):
        super().__init__(name_mapping)
        self.original_lower_bound = jnp.atleast_1d(original_lower_bound)
        self.original_upper_bound = jnp.atleast_1d(original_upper_bound)

        self.transform_func = lambda x: {
            name_mapping[1][i]: logit(
                (x[name_mapping[0][i]] - self.original_lower_bound[i])
                / (self.original_upper_bound[i] - self.original_lower_bound[i])
            )
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: (
                self.original_upper_bound[i] - self.original_lower_bound[i]
            )
            / (1 + jnp.exp(-x[name_mapping[1][i]]))
            + self.original_lower_bound[i]
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class SingleSidedUnboundTransform(BijectiveTransform):
    """
    Unbound upper limit transformation.

    Args:
        name_mapping (tuple[list[str], list[str]]): The name mapping between the input and output dictionary.
    """

    original_lower_bound: Float[Array, " n_dim"]

    def __repr__(self):
        return f"SingleSidedUnboundTransform(name_mapping={self.name_mapping}, original_lower_bound={self.original_lower_bound})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        original_lower_bound: Float | Float[Array, " n_dim"],
    ):
        super().__init__(name_mapping)
        self.original_lower_bound = jnp.atleast_1d(original_lower_bound)

        self.transform_func = lambda x: {
            name_mapping[1][i]: jnp.log(
                x[name_mapping[0][i]] - self.original_lower_bound[i]
            )
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.exp(x[name_mapping[1][i]])
            + self.original_lower_bound[i]
            for i in range(len(name_mapping[1]))
        }


@jaxtyped(typechecker=typechecker)
class PowerLawTransform(BijectiveTransform):
    """
    PowerLaw transformation.

    Args:
        name_mapping (tuple[list[str], list[str]]): The name mapping between the input and output dictionary.
    """

    xmin: Float
    xmax: Float
    alpha: Float

    def __repr__(self):
        return f"PowerLawTransform(name_mapping={self.name_mapping}, xmin={self.xmin}, xmax={self.xmax}, alpha={self.alpha})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        xmin: Float,
        xmax: Float,
        alpha: Float,
    ):
        super().__init__(name_mapping)
        self.xmin = xmin
        self.xmax = xmax
        self.alpha = alpha
        if alpha == -1.0:
            self.transform_func = lambda x: {
                name_mapping[1][i]: self.xmin
                * jnp.exp(x[name_mapping[0][i]] * jnp.log(self.xmax / self.xmin))
                for i in range(len(name_mapping[0]))
            }
            self.inverse_transform_func = lambda x: {
                name_mapping[0][i]: (
                    jnp.log(x[name_mapping[1][i]] / self.xmin)
                    / jnp.log(self.xmax / self.xmin)
                )
                for i in range(len(name_mapping[1]))
            }
        else:
            alphap1 = 1.0 + self.alpha
            self.transform_func = lambda x: {
                name_mapping[1][i]: (
                    self.xmin**alphap1
                    + x[name_mapping[0][i]] * (self.xmax**alphap1 - self.xmin**alphap1)
                )
                ** (1.0 / alphap1)
                for i in range(len(name_mapping[0]))
            }
            self.inverse_transform_func = lambda x: {
                name_mapping[0][i]: (
                    (x[name_mapping[1][i]] ** alphap1 - self.xmin**alphap1)
                    / (self.xmax**alphap1 - self.xmin**alphap1)
                )
                for i in range(len(name_mapping[1]))
            }


@jaxtyped(typechecker=typechecker)
class CartesianToPolarTransform(BijectiveTransform):
    """
    Transformation from (x, y) to (theta, r).

    Args:
        parameter_name (str): The name of the parameter to be transformed.
    """

    def __repr__(self):
        return f"CartesianToPolarTransform(name_mapping={self.name_mapping})"

    def __init__(
        self,
        parameter_name: str,
    ):
        super().__init__(
            name_mapping=(
                [f"{parameter_name}_x", f"{parameter_name}_y"],
                [f"{parameter_name}_theta", f"{parameter_name}_r"],
            )
        )
        self.transform_func = lambda x: {
            f"{parameter_name}_theta": jnp.arctan2(
                x[f"{parameter_name}_y"], x[f"{parameter_name}_x"]
            )
            + jnp.pi,
            f"{parameter_name}_r": jnp.sqrt(
                x[f"{parameter_name}_x"] ** 2 + x[f"{parameter_name}_y"] ** 2
            ),
        }
        self.inverse_transform_func = lambda x: {
            f"{parameter_name}_x": x[f"{parameter_name}_r"]
            * jnp.cos(x[f"{parameter_name}_theta"]),
            f"{parameter_name}_y": x[f"{parameter_name}_r"]
            * jnp.sin(x[f"{parameter_name}_theta"]),
        }


@jaxtyped(typechecker=typechecker)
class PeriodicTransform(BijectiveTransform):
    """Transform a periodic parameter onto a 2D circle.

    Maps ``(r, θ)`` — where ``θ ∈ [xmin, xmax]`` is the periodic angle and
    ``r`` is a scale — to Cartesian coordinates ``(x, y) = r(cos θ', sin θ')``
    (with ``θ' = 2π(θ - xmin)/(xmax - xmin)``).  The inverse recovers
    ``(r, θ)`` from ``(x, y)``.

    Attributes:
        xmin (Float): Lower bound of the periodic parameter.
        xmax (Float): Upper bound of the periodic parameter.
    """

    def __repr__(self):
        return f"PeriodicTransform(name_mapping={self.name_mapping}, xmin={self.xmin}, xmax={self.xmax})"

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        xmin: Float,
        xmax: Float,
    ):
        super().__init__(name_mapping)
        self.xmin = xmin
        self.xmax = xmax
        scaling = 2 * jnp.pi / (self.xmax - self.xmin)
        self.transform_func = lambda x: {
            f"{name_mapping[1][0]}": x[name_mapping[0][0]]
            * jnp.cos(scaling * (x[name_mapping[0][1]] - self.xmin)),
            f"{name_mapping[1][1]}": x[name_mapping[0][0]]
            * jnp.sin(scaling * (x[name_mapping[0][1]] - self.xmin)),
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][1]: self.xmin
            + (jnp.pi + jnp.arctan2(x[name_mapping[1][1]], x[name_mapping[1][0]]))
            / scaling,
            name_mapping[0][0]: jnp.sqrt(
                x[name_mapping[1][0]] ** 2 + x[name_mapping[1][1]] ** 2
            ),
        }


@jaxtyped(typechecker=typechecker)
class RayleighTransform(BijectiveTransform):
    """
    Transformation from Uniform(0, 1) to Rayleigh distribution with scale parameter sigma.

    Args:
        name_mapping (tuple[list[str], list[str]]): The name mapping between the input and output dictionary.
    """

    def __repr__(self):
        return (
            f"RayleighTransform(name_mapping={self.name_mapping}, sigma={self.sigma})"
        )

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        sigma: Float,
    ):
        super().__init__(name_mapping)
        self.sigma = sigma
        self.transform_func = lambda x: {
            name_mapping[1][i]: sigma * jnp.sqrt(-2 * jnp.log(x[name_mapping[0][i]]))
            for i in range(len(name_mapping[0]))
        }
        self.inverse_transform_func = lambda x: {
            name_mapping[0][i]: jnp.exp(-((x[name_mapping[1][i]] / sigma) ** 2) / 2)
            for i in range(len(name_mapping[1]))
        }


def reverse_bijective_transform(
    original_transform: BijectiveTransform,
) -> BijectiveTransform:
    """Construct the inverse of a bijective transform.

    Swaps the forward and inverse functions and the name mapping of
    ``original_transform``, returning a new :class:`BijectiveTransform` (or
    :class:`ConditionalBijectiveTransform`) whose forward direction is the
    original's inverse.

    Args:
        original_transform (BijectiveTransform): The transform to invert.

    Returns:
        BijectiveTransform: A new transform that is the reverse of the input.
    """
    reversed_name_mapping = (
        original_transform.name_mapping[1],
        original_transform.name_mapping[0],
    )
    if isinstance(original_transform, ConditionalBijectiveTransform):
        reversed_transform = ConditionalBijectiveTransform(
            name_mapping=reversed_name_mapping,
            conditional_names=original_transform.conditional_names,
        )
    else:
        reversed_transform = BijectiveTransform(name_mapping=reversed_name_mapping)
    reversed_transform.transform_func = original_transform.inverse_transform_func
    reversed_transform.inverse_transform_func = original_transform.transform_func
    reversed_transform.__repr__ = lambda: f"Reversed{repr(original_transform)}"

    return reversed_transform
