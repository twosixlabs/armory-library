import { Chart, Title, Tooltip, Legend, LineElement, PointElement, CategoryScale, LinearScale } from 'chart.js';
import { storeToRefs } from 'pinia';
import { computed } from 'vue';
import { Line } from "vue-chartjs";
import { useRuntimeSettings } from '../stores/runtime-settings.js';

const colors = [
    '#1e41dd',
    '#24d4a0',
    '#889ea9',
    '#002643',
];

export default {
    components: {
        Line,
    },
    props: {
        runs: Object,
    },
    setup(props) {
        Chart.register(Title, Tooltip, Legend, LineElement, PointElement, CategoryScale, LinearScale);

        const settings = storeToRefs(useRuntimeSettings());
        const { selectedMetric } = settings;

        const systemMetrics = computed(() => {
            const allMetrics = new Set();
            for (const run of props.runs) {
                for (const name of Object.keys(run.system_metrics)) {
                    allMetrics.add(name.slice(7));
                }
            }
            return [...allMetrics];
        });

        const chartOptions = computed(() => ({
            responsive: true,
            scales: {
                x: {
                    ticks: {
                        display: false,
                    },
                },
            },
        }));

        const chartData = computed(() => {
            if (!!selectedMetric.value) {
                const datasets = [];

                for (const [idx, run] of props.runs.entries()) {
                    const metrics = run.system_metrics["system/" + selectedMetric.value];
                    if (metrics == undefined) {
                        continue;
                    }
                    datasets.push({
                        label: run.info.run_name,
                        backgroundColor: colors[idx],
                        data: metrics.map((m) => m.value),
                    });
                }

                const max = Math.max(...datasets.map((d) => d.data.length));
                const labels = [...Array(max)].map((_, i) => i);

                return { labels, datasets };
            }
        });

        return { chartData, chartOptions, selectedMetric, systemMetrics };
    },
    template: `
        <div class="flex gap-2 items-center my-2">
            <span>System Metric</span>
            <select v-model="selectedMetric" class="select select-bordered select-sm w-60">
                <option disabled value="">Which metric?</option>
                <option v-for="metric in systemMetrics">
                    {{ metric }}
                </option>
            </select>
        </div>
        <div v-if="!!selectedMetric" class="flex justify-center h-[50vh] w-full">
            <Line :data="chartData" :options="chartOptions" class="bg-white shadow-md" />
        </div>
    `,
};
