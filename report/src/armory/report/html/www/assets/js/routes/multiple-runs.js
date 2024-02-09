import { useRoute } from 'vue-router';

export default {
    setup() {
        const route = useRoute();
        const runId = route.query.runs;
        return { runId };
    },
    template: `
        <p>comparing {{ runId }}</p>
    `,
};
