import { computed, watch } from 'vue';
import { RouterView, useRouter } from 'vue-router';
import Heading from '../components/heading.js';
import Tabs from '../components/tabs.js';
import { useEvaluationData } from '../stores/evaluation-data.js';

export default {
    props: {
        id: String,
    },
    components: {
        Heading,
        RouterView,
        Tabs,
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

        const tabs = [
            { dest: 'single-run-details', label: 'Details' },
            { dest: 'single-run-metrics', label: 'Metrics' },
            { dest: 'single-run-runtime', label: 'Runtime' },
            { dest: 'single-run-params', label: 'Parameters' },
            { dest: 'single-run-samples', label: 'Samples' },
            { dest: 'single-run-flowchart', label: 'Flowchart' },
        ];

        return { run, tabs };
    },
    template: `
        <div class="container">
            <heading>{{ run?.info.run_name }}</heading>
            <Tabs :tabs="tabs" />
            <router-view v-if="run" :run="run"></router-view>
        </div>
    `,
};
