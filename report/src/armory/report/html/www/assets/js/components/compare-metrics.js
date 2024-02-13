import { storeToRefs } from 'pinia';
import { computed } from 'vue';
import { RouterLink } from 'vue-router';
import { useMetricsSettings } from '../stores/metrics-settings.js';
import Button from './button.js';
import ChainColumnDropdown from './chain-column-dropdown.js';
import HiddenChainsDropdown from './hidden-chains-dropdown.js';
import HiddenMetricsDropdown from './hidden-metrics-dropdown.js';
import {
    BETTER_THAN_BASELINE,
    SAME_AS_BASELINE,
    WORSE_THAN_BASELINE,
    MetricCell,
} from './metric-cell.js';
import MetricColumnDropdown from './metric-column-dropdown.js';
import ParameterColumnDropdown from './parameter-column-dropdown.js';
import ParametersDropdown from './parameters-dropdown.js';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
    TableRowHeader,
} from './table.js';

const reorganizeMetrics = (runs, hiddenMetrics, hiddenChains) => {
    const byRunId = {};
    const allMetrics = new Set();
    const parameters = new Set();
    for (const run of runs) {
        for (const name of Object.keys(run.data.metrics)) {
            allMetrics.add(name);
        }
        for (const name of Object.keys(run.data.params)) {
            parameters.add(name);
        }
        byRunId[run.info.run_id] = run;
    }
    const columns = {};
    for (const key of allMetrics) {
        const segments = key.split("/");
        if (segments.length == 2 && segments[0] != "system") {
            const chain = segments[0];
            const metric = segments[1];
            if (hiddenMetrics.includes(metric) || hiddenChains.includes(chain)) {
                // skip
            } else if (metric in columns) {
                columns[metric].push(chain);
            } else {
                columns[metric] = [chain];
            }
        }
    }
    return { byRunId, columns, parameters };
};

export default {
    components: {
        Button,
        ChainColumnDropdown,
        HiddenChainsDropdown,
        HiddenMetricsDropdown,
        MetricCell,
        MetricColumnDropdown,
        ParameterColumnDropdown,
        ParametersDropdown,
        RouterLink,
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
        const {
            getMetricType,
            toggleBaselineRun,
        } = metricsSettings;
        const {
            baselineRun,
            precision,
            showParameters,
            hiddenChains,
            hiddenMetrics,
        } = storeToRefs(metricsSettings);

        const metrics = computed(() => 
            reorganizeMetrics(props.runs, hiddenMetrics.value, hiddenChains.value)
        );

        const getMetricValue = (run, chain, metric) => run.data.metrics[chain + "/" + metric];

        const compareToBaseline = (runId, chain, metric, value) => {
            if (!baselineRun.value || baselineRun.value == runId) {
                return SAME_AS_BASELINE
            }
            const baselineValue = metrics.value.byRunId[baselineRun.value].data.metrics[chain + "/" + metric];
            if (baselineValue == undefined) {
                return SAME_AS_BASELINE;
            }
            if (baselineValue < value) {
                return getMetricType(metric) == "high" ? BETTER_THAN_BASELINE : WORSE_THAN_BASELINE;
            }
            if (baselineValue > value) {
                return getMetricType(metric) == "high" ? WORSE_THAN_BASELINE : BETTER_THAN_BASELINE;
            }
            return SAME_AS_BASELINE;
        };

        return {
            baselineRun,
            compareToBaseline,
            getMetricValue,
            hiddenMetrics,
            metrics,
            precision,
            showParameters,
            toggleBaselineRun,
        };
    },
    template: `
        <div class="items-center flex flex-row gap-2 my-2">
            <HiddenMetricsDropdown></HiddenMetricsDropdown>
            <HiddenChainsDropdown></HiddenChainsDropdown>
            <ParametersDropdown :parameters="metrics.parameters"></ParametersDropdown>
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
                        class="border-x-2 border-white text-center"
                    >
                        <div class="items-center flex gap-2 justify-center">
                            {{ metric }}
                            <MetricColumnDropdown :metric="metric"></MetricColumnDropdown>
                        </div>
                    </TableHeader>
                    <TableHeader v-for="param in showParameters" :key="param"></TableHeader>
                    <TableHeader></TableHeader>
                </tr>
                <tr>
                    <TableHeader></TableHeader>
                    <template v-for="(chains, metric) in metrics.columns" :key="metric">
                        <TableHeader
                            v-for="chain in chains"
                            :key="chain"
                            class="[writing-mode:vertical-lr] border-x-2 border-white"
                        >
                            <div class="flex gap-2 justify-between">
                                {{ chain }}
                                <ChainColumnDropdown :chain="chain" class="[writing-mode:horizontal-tb]"></ChainColumnDropdown>
                            </div>
                        </TableHeader>
                    </template>
                    <TableHeader
                        v-for="param in showParameters"
                        :key="param"
                        class="[writing-mode:vertical-lr]"
                    >
                        <div class="flex gap-2 justify-between">
                            {{ param }}
                            <ParameterColumnDropdown
                                :parameter="param"
                                class="[writing-mode:horizontal-tb]"
                            ></ParameterColumnDropdown>
                        </div>
                    </TableHeader>
                    <TableHeader></TableHeader>
                </tr>
            </TableHead>
            <TableBody>
                <TableRow v-for="(run, runId) in metrics.byRunId" :key="runId">
                    <TableRowHeader>
                        <router-link
                            :to="{ name: 'single-run-metrics', params: { id: run.info.run_id } }"
                            class="text-twosix-blue"
                        >
                            {{ run.info.run_name }}
                        </router-link>
                    </TableRowHeader>
                    <template v-for="(chains, metric) in metrics.columns">
                        <MetricCell
                            v-for="chain in chains"
                            :comparison="compareToBaseline(runId, chain, metric, getMetricValue(run, chain, metric))"
                            :key="chain"
                            :precision="precision"
                            :value="getMetricValue(run, chain, metric)"
                        ></MetricCell>
                    </template>
                    <TableCell
                        v-for="param in showParameters"
                        :key="param"
                    >
                        {{ run.data.params[param] }}
                    </TableCell>
                    <TableCell>
                        <Button
                            :active="baselineRun == runId"
                            @click="toggleBaselineRun(runId)"
                        >
                            Baseline
                        </Button>
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    `,
};
