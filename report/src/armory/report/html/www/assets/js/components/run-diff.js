import { computed } from 'vue';
import { RouterLink } from 'vue-router';
import { useMetricsSettings } from '../stores/metrics-settings.js';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
    TableRowHeader,
} from './table.js';

const getMetrics = (runs) => {
    const allMetrics = new Set();
    for (const run of runs) {
        Object.keys(run.data.metrics).forEach(allMetrics.add, allMetrics);
    }
    const metrics = {};
    for (const key of allMetrics) {
        const segments = key.split("/");
        if (segments.length == 2 && segments[0] != "system") {
            const chain = segments[0];
            const metric = segments[1];
            if (metric in metrics) {
                metrics[metric].push(chain);
            } else {
                metrics[metric] = [chain];
            }
        }
    }
    return metrics;
};

const getParameters = (runs) => {
    const params = new Set();
    for (const run of runs) {
        Object.keys(run.data.params).forEach(params.add, params);
    }
    return [...params];
}

export default {
    components: {
        Table,
        TableBody,
        TableCell,
        TableHead,
        TableHeader,
        TableRow,
        TableRowHeader,
    },
    props: {
        runs: Object,
    },
    setup(props) {
        const metrics = computed(() => getMetrics(props.runs));
        const parameters = computed(() => getParameters(props.runs));
        return { metrics, parameters };
    },
    template: `
        <Table>
            <TableHead>
                <tr>
                    <TableHeader></TableHeader>
                    <TableHeader
                        v-for="run in runs"
                        :key="run.info.run_id"
                        class="[writing-mode:vertical-lr]"
                    >
                        {{ run.info.run_name }}
                    </TableHeader>
                </tr>
            </TableHead>
            <TableBody>
                <TableRow>
                    <TableRowHeader :colspan="runs.length + 1">
                        Metrics
                    </TableRowHeader>
                </TableRow>
                <template v-for="(chains, metric) in metrics" :key="metric">
                    <TableRow>
                        <TableRowHeader :colspan="runs.length + 1">
                            <span class="ml-4">
                                {{ metric }}
                            </span>
                        </TableRowHeader>
                    </TableRow>
                    <TableRow v-for="chain in chains">
                        <TableRowHeader>
                            <span class="ml-8">
                                {{ chain }}
                            </span>
                        </TableRowHeader>
                        <TableCell
                            v-for="run in runs"
                            :key="run.info.run_id"
                        >
                            {{ run.data.metrics[chain + "/" + metric].toFixed(3) }}
                        </TableCell>
                    </TableRow>
                </template>
                <TableRow>
                    <TableRowHeader :colspan="runs.length + 1">
                        Parameters
                    </TableRowHeader>
                </TableRow>
                <TableRow
                    v-for="parameter in parameters"
                    :key="parameter"
                >
                    <TableRowHeader>
                        <span class="ml-4">
                            {{ parameter }}
                        </span>
                    </TableRowHeader>
                    <TableCell
                        v-for="run in runs"
                        :key="run.info.run_id"
                        class="max-w-60 break-words"
                    >
                        {{ run.data.params[parameter] }}
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    `,
};
