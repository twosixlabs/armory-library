/**
 * @param {Record<string, any>} params 
 * @param {Record<string, any>} options
 */
export const setSearchParams = (params, options = {}) => {
    const newUrl = new URL(window.location);
    Object.entries(params).forEach(([k, v]) => newUrl.searchParams.set(k, v));
    if (options.replace) {
        window.history.replaceState({}, '', newUrl);
    } else {
        window.history.pushState({}, '', newUrl);
    }
};

export const getSearchParams = () => {
    return new URLSearchParams(window.location.search);
}
