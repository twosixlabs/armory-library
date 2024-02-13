import { storeToRefs } from 'pinia';
import { computed } from 'vue';
import { useMetricsSettings } from '../stores/metrics-settings.js';
import Button from './button.js';
import HiddenMetricsDropdown from './hidden-metrics-dropdown.js';
import { ChevronDownIcon } from './icons.js';
import {
    BETTER_THAN_BASELINE,
    SAME_AS_BASELINE,
    WORSE_THAN_BASELINE,
    MetricCell,
} from './metric-cell.js';
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

export default {
    components: {
        Button,
        ChevronDownIcon,
        HiddenMetricsDropdown,
        MetricCell,
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
        run: Object,
    },
    setup(props) {
        const metricsSettings = useMetricsSettings();
        const {
            getMetricType,
            toggleBaselineChain,
        } = metricsSettings;
        const {
            baselineChain,
            hiddenMetrics,
            precision,
        } = storeToRefs(metricsSettings);

        const [metricsByChain, allMetrics] = reorganizeMetrics(props.run.data.metrics);

        const visibleMetrics = computed(() => {
            if (hiddenMetrics.value) {
                return [...allMetrics].filter((m) => !hiddenMetrics.value.includes(m));
            }
            return allMetrics;
        });

        const compareToBaseline = (chain, metric, value) => {
            if (!baselineChain.value || baselineChain.value == chain) {
                return SAME_AS_BASELINE;
            }
            const baselineValue = metricsByChain[baselineChain.value][metric];
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
            baselineChain,
            compareToBaseline,
            metricsByChain,
            precision,
            toggleBaselineChain,
            visibleMetrics,
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
                    <TableHeader>Chain</TableHeader>
                    <TableHeader v-for="metric in visibleMetrics" :key="metric">
                        <div class="items-center flex gap-2">
                            {{ metric }}
                            <MetricColumnDropdown :metric="metric"></MetricColumnDropdown>
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
                            :active="baselineChain == entry[0]"
                            @click="toggleBaselineChain(entry[0])"
                        >
                            Baseline
                        </Button>
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    `,
};
