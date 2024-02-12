import { storeToRefs } from 'pinia';
import { useMetricsSettings } from '../stores/metrics-settings.js';
import Button from './button.js';
import { ChevronDownIcon } from './icons.js';

export default {
    components: {
        Button,
        ChevronDownIcon,
    },
    setup() {
        const settings = useMetricsSettings();
        const { hiddenMetrics } = storeToRefs(settings);
        return { hiddenMetrics, toggleMetric: settings.toggleMetric };
    },
    template: `
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
    `,
};
