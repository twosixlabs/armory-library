import { useRoute } from 'vue-router';

export default {
    setup() {
        const route = useRoute();
        const runId = route.params.id;
        return { runId };
    },
    template: `
        <p>single run {{ runId }}</p>
    `,
};
