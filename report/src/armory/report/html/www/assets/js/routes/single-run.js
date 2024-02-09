import { computed, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import Heading from '../components/heading.js';
import { useEvaluationData } from '../stores/evaluation-data.js';

export default {
    components: {
        Heading,
    },
    setup() {
        const route = useRoute();
        const router = useRouter();

        const evaluationData = useEvaluationData();
        const run = computed(() => evaluationData.runs.filter(
            (run) => run.info.run_id == route.params.id
        )[0]);
        watch(run, (newRun, oldRun) => {
            if (!newRun) {
                router.push({ path: '/' });
            }
        });

        return { run };
    },
    template: `
        <div class="container">
            <heading>{{ run?.info.run_name }}</heading>
        </div>
    `,
};
