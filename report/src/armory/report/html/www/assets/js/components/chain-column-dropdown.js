import { useMetricsSettings } from '../stores/metrics-settings.js';
import Button from './button.js';
import { ChevronDownIcon } from './icons.js';

export default {
    props: {
        chain: String,
    },
    components: {
        Button,
        ChevronDownIcon,
    },
    setup() {
        const settings = useMetricsSettings();
        const { toggleChain } = settings;
        return { toggleChain };
    },
    template: `
        <div class="dropdown">
            <Button minimal tabindex="0">
                <ChevronDownIcon></ChevronDownIcon>
            </Button>
            <ul tabindex="0" class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-52">
                <li>
                    <a @click="toggleChain(chain)">
                        Hide
                    </a>
                </li>
            </ul>
        </div>
    `,
};
