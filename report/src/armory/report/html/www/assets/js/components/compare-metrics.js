import { storeToRefs } from 'pinia';
import { computed } from 'vue';
import { useMetricsSettings } from '../stores/metrics-settings.js';
import HiddenMetricsDropdown from './hidden-metrics-dropdown.js';
import MetricColumnDropdown from './metric-column-dropdown.js';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
    TableRowHeader,
} from './table.js';

const reorganizeMetrics = (runs, hiddenMetrics) => {
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
            if (hiddenMetrics.includes(metric)) {
                // skip
            } else if (metric in columns) {
                columns[metric].push(chain);
            } else {
                columns[metric] = [chain];
            }
        }
    }
    return { byRun, columns };
};

export default {
    components: {
        HiddenMetricsDropdown,
        MetricColumnDropdown,
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
        const { precision, hiddenMetrics } = storeToRefs(metricsSettings);

        const metrics = computed(() => reorganizeMetrics(props.runs, hiddenMetrics.value));

        return {
            hiddenMetrics,
            metrics,
            precision,
        };
    },
    template: `
        <div class="items-center flex flex-row gap-2 my-2">
            <HiddenMetricsDropdown></HiddenMetricsDropdown>
            <span class="border-l-2 pl-2">
                Precision
            </span>
            <input
                v-model="precision"
                class="appearance-none border border-zinc-300 focus:border-zinc-400 focus:outline-none leading-6 pl-3 pr-2 py-1.5 rounded-md w-20"
                min="0"
                max="9"
                type="number"
            />
        </div>
        <Table>
            <TableHead>
                <tr>
                    <TableHeader class="text-center">Run</TableHeader>
                    <TableHeader
                        v-for="(chains, metric) in metrics.columns"
                        :key="metric"
                        :colspan="chains.length"
                        class="border-l-2 border-white text-center"
                    >
                        <div class="items-center flex gap-2 justify-center">
                            {{ metric }}
                            <MetricColumnDropdown :metric="metric"></MetricColumnDropdown>
                        </div>
                    </TableHeader>
                </tr>
                <tr>
                    <TableHeader></TableHeader>
                    <template v-for="(chains, metric) in metrics.columns" :key="metric">
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
                <TableRow v-for="(runMetrics, runName) in metrics.byRun" :key="runName">
                    <TableRowHeader>
                        {{ runName }}
                    </TableRowHeader>
                    <template v-for="(chains, metric) in metrics.columns">
                        <TableCell v-for="chain in chains" :key="chain">
                            {{ runMetrics[chain + "/" + metric].toFixed(precision) }}
                        </TableCell>
                    </template>
                </TableRow>
            </TableBody>
        </Table>
    `,
};
