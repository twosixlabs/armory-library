import { computed, watch } from 'vue';
import { RouterLink, RouterView, useRouter } from 'vue-router';
import Heading from '../components/heading.js';
import { useEvaluationData } from '../stores/evaluation-data.js';

export default {
    components: {
        Heading,
        RouterLink,
        RouterView,
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

        return { runs, runIds };
    },
    template: `
        <div class="container">
            <heading>
                Comparing {{ runs.length }} runs
            </heading>
            <div role="tablist" class="tabs tabs-lifted">
                <router-link
                    :to="{ name: 'compare-runs-metrics', query: { runs: runIds } }"
                    role="tab"
                    class="tab"
                    active-class="tab-active"
                >
                    Metrics
                </router-link>
                <router-link
                    :to="{ name: 'compare-runs-diff', query: { runs: runIds } }"
                    role="tab"
                    class="tab"
                    active-class="tab-active"
                >
                    Diff
                </router-link>
            </div>
            <router-view v-if="runs.length" :runs="runs"></router-view>
        </div>
    `,
};
