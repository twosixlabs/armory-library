import { computed } from 'vue';
import { useEvaluationData } from '../stores/evaluation-data.js';
import Heading from './heading.js';

const Field = {
    props: {
        title: String,
    },
    template: `
        <dt class="uppercase">
            {{ title }}
        </dt>
        <dd class="ml-2">
            <slot></slot>
        </dd>
    `,
};

export default {
    components: {
        Field,
        Heading,
    },
    setup() {
        const evaluationData = useEvaluationData();

        const title = computed({
            get() {
                const runs = evaluationData.selectedRuns;
                if (runs.length > 1) {
                    return `${runs.length} runs selected`;
                }
                if (runs.length == 1) {
                    return runs[0].info.run_name;
                }
                return "No runs selected";
            }
        });

        const info = computed({
            get() {
                const runs = evaluationData.selectedRuns;
                if (runs.length == 1) {
                    return runs[0].info;
                }
                return null;
            }
        });

        return { info, title };
    },
    template: `
        <heading>{{ title }}</heading>
        <dl>
            <Field title="Run ID">
                {{ info?.run_id }}
            </Field>
            <Field title="Status">
                {{ info?.status }}
            </Field>
            <Field title="Started">
                {{ info?.start_time }}
            </Field>
            <Field title="Duration">
                {{ info && info.end_time - info.start_time }}
            </Field>
        </dl>
    `,
};
