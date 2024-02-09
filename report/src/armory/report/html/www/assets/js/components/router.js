import { storeToRefs } from 'pinia';
import overview from '../routes/overview.js';
import { useRoute } from '../stores/route.js';
import metrics from './metrics.js';
import parameters from './parameters.js';
import summary from './summary.js';

export default {
    setup() {
        const store = useRoute();
        const { route: currentRoute } = storeToRefs(store);

        const routes = {
            overview,
            metrics,
            parameters,
            summary,
        };

        return { currentRoute, routes };
    },
    template: `
        <component :is="routes[currentRoute]"></component>
    `,
};
