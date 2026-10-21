"""Dependency-free checks for same-machine CPU/CUDA results."""
import math


def compare_values(reference, actual, *, atol, rtol):
    if not all(math.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError('Tolerances must be finite and nonnegative')
    errors = []
    count = 0
    max_absolute_error = 0.0

    def fail(path, reason):
        if len(errors) < 20:
            errors.append(f'{path}: {reason}')

    def visit(left, right, path):
        nonlocal count, max_absolute_error
        if type(left) is not type(right):
            fail(path, 'types differ')
        elif isinstance(left, dict):
            if left.keys() != right.keys():
                fail(path, 'keys differ')
            else:
                for key in left:
                    visit(left[key], right[key], f'{path}.{key}')
        elif isinstance(left, list):
            if len(left) != len(right):
                fail(path, 'lengths differ')
            else:
                for i, (a, b) in enumerate(zip(left, right)):
                    visit(a, b, f'{path}[{i}]')
        elif isinstance(left, float):
            count += 1
            if not math.isfinite(left) or not math.isfinite(right):
                fail(path, 'non-finite number')
                return
            error = abs(left - right)
            max_absolute_error = max(max_absolute_error, error)
            if error > atol + rtol * abs(left):
                fail(path, f'{right} differs from CPU {left} by {error}')
        elif left != right:
            fail(path, 'values differ')

    visit(reference, actual, 'result')
    if count == 0:
        fail('result', 'no numerical observations')
    return {'passed': not errors, 'errors': errors,
            'numbers_compared': count, 'max_absolute_error': max_absolute_error}


def compare_results(cpu, cuda, *, atol, rtol):
    for result, expected in ((cpu, 'cpu'), (cuda, 'cuda')):
        if result.get('backend') != expected:
            raise ValueError(f'Expected {expected} result')
        if not result.get('metadata') or not result.get('values'):
            raise ValueError('Missing metadata or observations')
    if cpu['metadata'] != cuda['metadata']:
        raise ValueError('Input/model/source metadata differ')
    return compare_values(cpu['values'], cuda['values'], atol=atol, rtol=rtol)
