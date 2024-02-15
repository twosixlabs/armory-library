export function argmax(values) {
    return values.map(
        (val, idx) => [val, idx]
    ).reduce(
        (acc, curr) => (curr[0] > acc[0] ? curr : acc)
    )[1];
}
