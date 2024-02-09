import { defineStore } from 'pinia';
import { getSearchParams, setSearchParams } from '../utils/history.js';

const ROUTE_KEY = "route";
const DEFAULT_ROUTE = "overview";

const getInitialState = () => {
    const params = getSearchParams();

    return {
        route: params.has(ROUTE_KEY) ? params.get(ROUTE_KEY) : DEFAULT_ROUTE,
    };
};

export const useRoute = defineStore('route', {
    state: getInitialState,
    actions: {
        clearRoute() {
            this.setRoute(DEFAULT_ROUTE);
        },
        setRoute(route) {
            this.route = route;
            setSearchParams({ [ROUTE_KEY]: route }, { replace: false });
        },
    }
})