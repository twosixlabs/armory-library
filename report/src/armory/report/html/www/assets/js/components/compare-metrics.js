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

const reorganizeMetrics = (runs) => {
    const byRun = {};
    const allMetrics = new Set();
    for (const run of runs) {
        for (const [name, value] of Object.entries(run.data.metrics)) {
            allMetrics.add(name);
        }
        byRun[run.info.run_name] = run.data.metrics;
    }
    const columns = {};
    for (const key of allMetrics) {
        const segments = key.split("/");
        if (segments.length == 2 && segments[0] != "system") {
            const chain = segments[0];
            const metric = segments[1];
            if (metric in columns) {
                columns[metric].push(chain);
            } else {
                columns[metric] = [chain];
            }
        }
    }
    return [byRun, columns];
};

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
        const metricsSettings = useMetricsSettings();

        const [metricsByRun, columns] = reorganizeMetrics(props.runs);

        return { columns, metricsByRun };
    },
    template: `
        <Table>
            <TableHead>
                <tr>
                    <TableHeader class="text-center">Run</TableHeader>
                    <TableHeader
                        v-for="(chains, metric) in columns"
                        :key="metric"
                        :colspan="chains.length"
                        class="border-l-2 border-white text-center"
                    >
                        {{ metric }}
                    </TableHeader>
                </tr>
                <tr>
                    <TableHeader></TableHeader>
                    <template v-for="(chains, metric) in columns" :key="metric">
                        <TableHeader
                            v-for="chain in chains"
                            :key="chain"
                            class="[writing-mode:vertical-lr] border-l-2 border-white"
                        >
                            {{ chain }}
                        </TableHeader>
                    </template>
                </tr>
            </TableHead>
            <TableBody>
                <TableRow v-for="(runMetrics, runName) in metricsByRun" :key="runName">
                    <TableRowHeader>
                        {{ runName }}
                    </TableRowHeader>
                    <template v-for="(chains, metric) in columns">
                        <TableCell v-for="chain in chains" :key="chain">
                            {{ runMetrics[chain + "/" + metric].toFixed(3) }}
                        </TableCell>
                    </template>
                </TableRow>
            </TableBody>
        </Table>
    `,
};
