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
        const { hiddenChains } = storeToRefs(settings);
        return { hiddenChains, toggleChain: settings.toggleChain };
    },
    template: `
        <div class="dropdown">
            <Button :disabled="hiddenChains.length == 0" tabindex="0">
                Chains
                <ChevronDownIcon></ChevronDownIcon>
            </Button>
            <ul tabindex="0" class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-52">
                <li v-for="chain in hiddenChains" :key="chain">
                    <a @click="toggleChain(chain)">{{ chain }}</a>
                </li>
            </ul>
        </div>
    `,
};
