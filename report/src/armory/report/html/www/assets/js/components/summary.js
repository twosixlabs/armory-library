import dayjs from 'dayjs';
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
        <dd class="ml-4">
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

        const formatDuration = (duration) => dayjs.duration(duration).format('HH:mm:ss');
        const formatTime = (time) => `${dayjs(time).toISOString()} (${dayjs(time).fromNow()})`;

        return { formatDuration, formatTime, info, title };
    },
    template: `
        <heading class="mb-2">{{ title }}</heading>
        <dl>
            <Field title="Run ID">
                {{ info?.run_id }}
            </Field>
            <Field title="Status">
                {{ info?.status }}
            </Field>
            <Field title="Started">
                {{ info && formatTime(info.start_time) }}
            </Field>
            <Field title="Duration">
                {{ info && formatDuration(info.end_time - info.start_time) }}
            </Field>
        </dl>
    `,
};
