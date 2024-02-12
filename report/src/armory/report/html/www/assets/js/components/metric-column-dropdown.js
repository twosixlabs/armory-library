import { useMetricsSettings } from '../stores/metrics-settings.js';
import Button from './button.js';
import { ChevronDownIcon } from './icons.js';

export default {
    props: {
        metric: String,
    },
    components: {
        Button,
        ChevronDownIcon,
    },
    setup() {
        const settings = useMetricsSettings();
        const { setMetricType, toggleMetric } = settings;
        return { setMetricType, toggleMetric };
    },
    template: `
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
    `,
};
