import { Chart, Title, Tooltip, Legend, LineElement, PointElement, CategoryScale, LinearScale } from 'chart.js';
import dayjs from 'dayjs';
import { storeToRefs } from 'pinia';
import { computed } from 'vue';
import { Line } from "vue-chartjs";
import { useRuntimeSettings } from '../stores/runtime-settings.js';
import { formatTime } from '../utils/format.js';

export default {
    components: {
        Line,
    },
    props: {
        run: Object,
    },
    setup(props) {
        Chart.register(Title, Tooltip, Legend, LineElement, PointElement, CategoryScale, LinearScale);

        const settings = storeToRefs(useRuntimeSettings());
        const { selectedMetric } = settings;

        const systemMetrics = computed(() => 
            Object.keys(props.run.system_metrics).map((key) => key.slice(7)));

        const chartOptions = computed(() => ({
            responsive: true,
            scales: {
                x: {
                    type: 'linear',
                    ticks: {
                        display: true,
                        callback: function(value, index, values) {
                            return dayjs(value).format("YYYY-MM-DD HH:mm:ss");
                        },
                        stepSize: 60000
                    },
                },
            },
        }));

        const chartData = computed(() => {
            if (!!selectedMetric.value) {
                const metrics = props.run.system_metrics["system/" + selectedMetric.value];
                if (metrics == undefined) {
                    return { datasets: [] };
                }
                return {
                    labels: metrics.map((m) => m.timestamp),
                    datasets: [
                        {
                            label: selectedMetric.value,
                            backgroundColor: '#1e41dd',
                            data: metrics.map((m) => m.value),
                        },
                    ],
                };
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
        <div v-if="!!selectedMetric" class="flex justify-center w-full">
            <Line :data="chartData" :options="chartOptions" />
        </div>
    `,
};
