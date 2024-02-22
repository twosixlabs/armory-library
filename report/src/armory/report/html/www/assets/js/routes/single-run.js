import { computed, watch } from 'vue';
import { RouterLink, RouterView, useRouter } from 'vue-router';
import Heading from '../components/heading.js';
import { useEvaluationData } from '../stores/evaluation-data.js';

export default {
    props: {
        id: String,
    },
    components: {
        Heading,
        RouterLink,
        RouterView,
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
            <div role="tablist" class="tabs tabs-lifted">
                <router-link
                    v-for="tab in tabs"
                    :key="tab.dest"
                    :to="{ name: tab.dest }"
                    role="tab"
                    class="tab"
                    exact-active-class="tab-active"
                >
                    {{ tab.label }}
                </router-link>
            </div>
            <router-view v-if="run" :run="run"></router-view>
        </div>
    `,
};
