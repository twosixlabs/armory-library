import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import Button from '../components/button.js';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
    TableRowHeader,
} from '../components/table.js';

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
                "text-green-700": props.comparison > 0,
                "text-red-700": props.comparison < 0,
            },
            span: {
                "border-r-[1rem]": props.comparison != 0,
                "border-green-700": props.comparison > 0,
                "border-red-700": props.comparison < 0,
                "pr-1": props.comparison != 0,
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

        const compareToBaseline = (chain, metric, value) => {
            if (!baseline.value || baseline.value == chain) {
                return 0;
            }
            const baselineValue = metricsByChain[baseline.value][metric];
            if (baselineValue == undefined) {
                return 0;
            }
            if (baselineValue < value) return 1;
            if (baselineValue > value) return -1;
            return 0;
        };

        return {
            allMetrics,
            baseline,
            compareToBaseline,
            metricsByChain,
            precision,
            toggleBaseline,
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
                    <TableHeader v-for="metric in allMetrics" :key="metric">
                        {{ metric }}
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
                        v-for="metric in allMetrics"
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
