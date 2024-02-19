import { storeToRefs } from 'pinia';
import { computed, ref } from 'vue';
import Button from '../components/button.js';
import { ChevronDownIcon, ChevronUpIcon } from '../components/icons.js';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
    TableRowHeader,
} from '../components/table.js';
import { useMetricsSettings } from '../stores/metrics-settings.js';

const getMetrics = (runs) => {
    const allMetrics = new Set();
    for (const run of runs) {
        Object.keys(run.data.metrics).forEach(allMetrics.add, allMetrics);
    }
    const metrics = {};
    for (const key of allMetrics) {
        const segments = key.split("/");
        if (segments.length >= 2 && segments[0] != "system") {
            const chain = segments[0];
            const metric = segments.slice(1).join("/");
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
        Button,
        ChevronDownIcon,
        ChevronUpIcon,
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
        const { precision, showAll } = storeToRefs(metricsSettings);

        const metrics = computed(() => getMetrics(props.runs));
        const collapseMetrics = ref(false);
        const getMetricRowClass = (chain, metric) => {
            const values = new Set();
            for (const run of props.runs) {
                const value = run.data.metrics[chain + "/" + metric];
                if (value != undefined) {
                    values.add(value.toFixed(metricsSettings.precision))
                } else {
                    values.add(null);
                }
            }

            const num = [...values].length;
            return {
                'even:bg-zinc-50': num <= 1,
                'bg-twosix-green': num > 1,
                'tw-collapse': collapseMetrics.value || (!showAll.value && num <= 1),
            };
        };

        const parameters = computed(() => getParameters(props.runs));
        const collapseParams = ref(false);
        const getParamRowClass = (param) => {
            const values = new Set();
            for (const run of props.runs) {
                values.add(run.data.params[param]);
            }

            const num = [...values].length;
            return {
                'even:bg-zinc-50': num <= 1,
                'bg-twosix-green': num > 1,
                'tw-collapse': collapseParams.value || (!showAll.value && num <= 1),
            };
        }

        return {
            collapseMetrics,
            collapseParams,
            getMetricRowClass,
            getParamRowClass,
            metrics,
            parameters,
            precision,
            showAll,
        };
    },
    template: `
        <div class="items-center flex flex-row gap-2 my-2">
            <span>
                Precision
            </span>
            <input
                v-model="precision"
                class="appearance-none border border-zinc-300 focus:border-zinc-400 focus:outline-none leading-6 pl-3 pr-2 py-1.5 rounded-md w-20"
                min="0"
                max="9"
                type="number"
            />
            <input
                v-model="showAll"
                id="show-all"
                type="checkbox"
            />
            <label for="show-all" class="hover:cursor-pointer">
                Show all
            </label>
        </div>
        <Table>
            <TableHead>
                <tr>
                    <TableHeader></TableHeader>
                    <TableHeader
                        v-for="run in runs"
                        :key="run.info.run_id"
                    >
                        <router-link
                            :to="{ name: 'single-run', params: { id: run.info.run_id } }"
                            class="text-twosix-blue"
                        >
                            {{ run.info.run_name }}
                        </router-link>
                    </TableHeader>
                </tr>
            </TableHead>
            <TableBody>
                <tr class="tw-collapse">
                    <td>hello</td>
                    <td>to</td>
                    <td>you</td>
                </tr>
                <TableRow>
                    <TableRowHeader :colspan="runs.length + 1">
                        <div class="flex items-center">
                            Metrics
                            <Button @click="collapseMetrics = !collapseMetrics" minimal class="ml-2">
                                <ChevronDownIcon v-if="collapseMetrics" />
                                <ChevronUpIcon v-if="!collapseMetrics" />
                            </Button>
                        </div>
                    </TableRowHeader>
                </TableRow>
                <template v-for="(chains, metric) in metrics" :key="metric">
                    <TableRow :class="{'tw-collapse': collapseMetrics}">
                        <TableRowHeader :colspan="runs.length + 1">
                            <span class="ml-4">
                                {{ metric }}
                            </span>
                        </TableRowHeader>
                    </TableRow>
                    <tr
                        v-for="chain in chains"
                        :class="getMetricRowClass(chain, metric)"
                    >
                        <TableRowHeader>
                            <span class="ml-8">
                                {{ chain }}
                            </span>
                        </TableRowHeader>
                        <TableCell
                            v-for="run in runs"
                            :key="run.info.run_id"
                        >
                            {{ run.data.metrics[chain + "/" + metric]?.toFixed(precision) }}
                        </TableCell>
                    </tr>
                </template>
                <TableRow>
                    <TableRowHeader :colspan="runs.length + 1">
                        <div class="flex items-center">
                            Parameters
                            <Button @click="collapseParams = !collapseParams" minimal class="ml-2">
                                <ChevronDownIcon v-if="collapseParams" />
                                <ChevronUpIcon v-if="!collapseParams" />
                            </Button>
                        </div>
                    </TableRowHeader>
                </TableRow>
                <tr
                    v-for="parameter in parameters"
                    :key="parameter"
                    :class="getParamRowClass(parameter)"
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
                </tr>
            </TableBody>
        </Table>
    `,
};
