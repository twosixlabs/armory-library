import { storeToRefs } from 'pinia';
import { computed } from 'vue';
import { useMetricsSettings } from '../stores/metrics-settings.js';
import Button from './button.js';
import { ChevronDownIcon } from './icons.js';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
    TableRowHeader,
} from './table.js';

const BETTER_THAN_BASELINE = 1;
const SAME_AS_BASELINE = 0;
const WORSE_THAN_BASELINE = -1;

const reorganizeMetrics = (flatMetrics) => {
    const byChain = {};
    const allMetrics = new Set();
    for (const [key, value] of Object.entries(flatMetrics)) {
        const segments = key.split("/");
        if (segments.length == 2 && segments[0] != "system") {
            const chain = segments[0];
            const metric = segments[1];
            allMetrics.add(metric);
            if (chain in byChain) {
                byChain[chain][metric] = value;
            } else {
                byChain[chain] = { [metric]: value };
            }
        }
    }
    return [byChain, allMetrics];
};

const MetricCell = {
    components: {
        TableCell,
    },
    props: {
        comparison: Number,
        precision: Number,
        value: Number,
    },
    setup(props) {
        const classes = computed(() => ({
            cell: {
                "text-green-700": props.comparison == BETTER_THAN_BASELINE,
                "text-red-700": props.comparison == WORSE_THAN_BASELINE,
            },
            span: {
                "border-r-[1rem]": props.comparison != SAME_AS_BASELINE,
                "border-green-700": props.comparison == BETTER_THAN_BASELINE,
                "border-red-700": props.comparison == WORSE_THAN_BASELINE,
                "pr-1": props.comparison != SAME_AS_BASELINE,
            },
        }));
        return { classes };
    },
    template: `
        <TableCell :class="classes.cell">
            <span :class="classes.span">
                {{ value.toFixed(precision) }}
            </span>
        </TableCell>
    `,
};

export default {
    components: {
        Button,
        ChevronDownIcon,
        MetricCell,
        Table,
        TableBody,
        TableCell,
        TableHead,
        TableHeader,
        TableRow,
        TableRowHeader,
    },
    props: {
        metrics: Object,
    },
    setup(props) {
        const metricsSettings = useMetricsSettings();
        const {
            getMetricType,
            setMetricType,
            toggleBaseline,
            toggleMetric,
        } = metricsSettings;
        const {
            baseline,
            hiddenMetrics,
            precision,
        } = storeToRefs(metricsSettings);

        const [metricsByChain, allMetrics] = reorganizeMetrics(props.metrics);

        const visibleMetrics = computed(() => {
            if (hiddenMetrics.value) {
                return [...allMetrics].filter((m) => !hiddenMetrics.value.includes(m));
            }
            return allMetrics;
        });

        const compareToBaseline = (chain, metric, value) => {
            if (!baseline.value || baseline.value == chain) {
                return SAME_AS_BASELINE;
            }
            const baselineValue = metricsByChain[baseline.value][metric];
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
            baseline,
            compareToBaseline,
            hiddenMetrics,
            metricsByChain,
            precision,
            setMetricType,
            toggleBaseline,
            toggleMetric,
            visibleMetrics,
        };
    },
    template: `
        {{ hiddenMetrics }}
        <div class="items-center flex flex-row gap-2 my-2">
            <div class="dropdown">
                <Button :disabled="hiddenMetrics.length == 0" tabindex="0">
                    Columns
                    <ChevronDownIcon></ChevronDownIcon>
                </Button>
                <ul tabindex="0" class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-52">
                    <li v-for="metric in hiddenMetrics" :key="metric">
                        <a @click="toggleMetric(metric)">{{ metric }}</a>
                    </li>
                </ul>
            </div>
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
                    <TableHeader>Chain</TableHeader>
                    <TableHeader v-for="metric in visibleMetrics" :key="metric">
                        <div class="items-center flex gap-2">
                            {{ metric }}
                            <div class="dropdown">
                                <Button minimal tabindex="0">
                                    <ChevronDownIcon></ChevronDownIcon>
                                </Button>
                                <ul tabindex="0" class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-52">
                                    <li>
                                        <a @click="toggleMetric(metric)">
                                            Hide
                                        </a>
                                    </li>
                                    <li>
                                        <a @click="setMetricType(metric, 'low')">
                                            Lower is better
                                        </a>
                                    </li>
                                    <li>
                                        <a @click="setMetricType(metric, 'high')">
                                            Higher is better
                                        </a>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </TableHeader>
                    <TableHeader></TableHeader>
                </tr>
            </TableHead>
            <TableBody>
                <TableRow v-for="entry in Object.entries(metricsByChain)" :key="entry[0]">
                    <TableRowHeader>
                        {{ entry[0] }}
                    </TableRowHeader>
                    <MetricCell
                        v-for="metric in visibleMetrics"
                        :comparison="compareToBaseline(entry[0], metric, entry[1][metric])"
                        :key="metric"
                        :precision="precision"
                        :value="entry[1][metric]"
                    ></MetricCell>
                    <TableCell>
                        <Button
                            :active="baseline == entry[0]"
                            @click="toggleBaseline(entry[0])"
                        >
                            Baseline
                        </Button>
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    `,
};
