import { computed, watch } from 'vue';
import { useRouter } from 'vue-router';
import Heading from '../components/heading.js';
import RunMetrics from '../components/run-metrics.js';
import { useEvaluationData } from '../stores/evaluation-data.js';

export default {
    props: {
        id: String,
    },
    components: {
        Heading,
        RunMetrics,
    },
    setup(props) {
        const router = useRouter();

        const evaluationData = useEvaluationData();
        const run = computed(() => evaluationData.runs.filter(
            (run) => run.info.run_id == props.id
        )[0]);
        watch(run, (newRun) => {
            if (!newRun) {
                router.push({ name: 'index' });
            }
        });

        return { run };
    },
    template: `
        <div class="container">
            <heading>{{ run?.info.run_name }}</heading>
            <run-metrics v-if="run" :metrics="run.data.metrics"></run-metrics>
        </div>
    `,
};
