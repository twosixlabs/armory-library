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

        return { run };
    },
    template: `
        <div class="container">
            <heading>{{ run?.info.run_name }}</heading>
            <div role="tablist" class="tabs tabs-lifted">
                <router-link
                    :to="{ name: 'single-run-metrics' }"
                    role="tab"
                    class="tab"
                    active-class="tab-active"
                >
                    Metrics
                </router-link>
                <router-link
                    :to="{ name: 'single-run-params' }"
                    role="tab"
                    class="tab"
                    active-class="tab-active"
                >
                    Parameters
                </router-link>
                <router-link
                    :to="{ name: 'single-run-pipeline' }"
                    role="tab"
                    class="tab"
                    active-class="tab-active"
                >
                    Pipeline
                </router-link>
            </div>
            <router-view v-if="run" :run="run"></router-view>
        </div>
    `,
};
