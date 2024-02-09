import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
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

const addTo = (maybeArray, value) => {
    if (Array.isArray(maybeArray)) {
        maybeArray.push(value);
        return maybeArray;
    }
    if (maybeArray) {
        return [maybeArray, value];
    }
    return [value];
}

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
        const route = useRoute();
        const router = useRouter();

        const baseline = computed({
            get() {
                return route.query.baseline;
            },
            set(baseline) {
                router.push({ query: { ...route.query, baseline }});
            },
        });
        
        const toggleBaseline = (chain) => {
            if (baseline.value == chain) {
                baseline.value = "";
            } else {
                baseline.value = chain;
            }
        };

        const precision = computed({
            get() {
                return route.query.precision ? Number.parseInt(route.query.precision) : 3;
            },
            set(precision) {
                router.push({ query: { ...route.query, precision } });
            },
        });

        const [metricsByChain, allMetrics] = reorganizeMetrics(props.metrics);

        const visibleMetrics = computed(() => {
            if (route.query.hide) {
                return [...allMetrics].filter((m) => !route.query.hide.includes(m));
            }
            return allMetrics;
        });

        const hideMetric = (metric) => {
            const hide = addTo(route.query.hide, metric);
            hide.push(metric);
            router.push({ query: { ...route.query, hide } });
        };

        const compareToBaseline = (chain, metric, value) => {
            if (!baseline.value || baseline.value == chain) {
                return SAME_AS_BASELINE;
            }
            const baselineValue = metricsByChain[baseline.value][metric];
            if (baselineValue == undefined) {
                return SAME_AS_BASELINE;
            }
            if (baselineValue < value) {
                return route.query[`metric.${metric}`] == "high" ? BETTER_THAN_BASELINE : WORSE_THAN_BASELINE;
            }
            if (baselineValue > value) {
                return route.query[`metric.${metric}`] == "high" ? WORSE_THAN_BASELINE : BETTER_THAN_BASELINE;
            }
            return SAME_AS_BASELINE;
        };

        const setMetricType = (metric, metricType) => {
            router.push({ query: { ...route.query, [`metric.${metric}`]: metricType }});
        };

        return {
            baseline,
            compareToBaseline,
            hideMetric,
            metricsByChain,
            precision,
            setMetricType,
            toggleBaseline,
            visibleMetrics,
        };
    },
    template: `
        <div class="my-2">
            <span class="mr-2">
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
                                        <a @click="hideMetric(metric)">
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
