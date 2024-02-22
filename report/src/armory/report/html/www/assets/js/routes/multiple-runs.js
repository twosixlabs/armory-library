import { computed, watch } from 'vue';
import { RouterLink, RouterView, useRouter } from 'vue-router';
import Heading from '../components/heading.js';
import Tabs from '../components/tabs.js';
import { useEvaluationData } from '../stores/evaluation-data.js';

export default {
    components: {
        Heading,
        RouterLink,
        RouterView,
        Tabs,
    },
    setup() {
        const router = useRouter();
        const runIds = computed(() => router.currentRoute.value.query.runs || []);
        watch(runIds, (newRunIds) => {
            if (newRunIds.length < 2) {
                router.push({ name: 'index' });
            }
        }, { immediate: true });

        const evaluationData = useEvaluationData();
        const runs = computed(() => evaluationData.runs.filter(
            (run) => runIds.value.includes(run.info.run_id)
        ));
        watch(runs, (newRuns) => {
            if (newRuns.length < 2) {
                router.push({ name: 'index' });
            }
        }, { immediate: true });

        const tabs = computed(() => ([
            {
                dest: { name: 'compare-runs-metrics', query: { runs: runIds.value } },
                label: 'Metrics',
            },
            {
                dest: { name: 'compare-runs-diff', query: { runs: runIds.value } },
                label: 'Diff',
            },
        ]));

        return { runs, runIds, tabs };
    },
    template: `
        <div class="container">
            <heading>
                Comparing {{ runs.length }} runs
            </heading>
            <Tabs :tabs="tabs" />
            <router-view v-if="runs.length" :runs="runs"></router-view>
        </div>
    `,
};
