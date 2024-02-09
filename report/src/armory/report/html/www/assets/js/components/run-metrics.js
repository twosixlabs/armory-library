import { computed } from 'vue';
import { useRoute, useRouter } from 'vue-router';
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
        metrics: Object,
    },
    setup(props) {
        const route = useRoute();
        const router = useRouter();

        const precision = computed({
            get() {
                return route.query.precision || 3;
            },
            set(precision) {
                router.push({ query: { precision } });
            },
        });

        const [metricsByChain, allMetrics] = reorganizeMetrics(props.metrics);

        return { allMetrics, metricsByChain, precision };
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
                </tr>
            </TableHead>
            <TableBody>
                <TableRow v-for="entry in Object.entries(metricsByChain)" :key="entry[0]">
                    <TableRowHeader>
                        {{ entry[0] }}
                    </TableRowHeader>
                    <TableCell v-for="metric in allMetrics" :key="metric">
                        {{ entry[1][metric].toFixed(precision) }}
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    `,
};
