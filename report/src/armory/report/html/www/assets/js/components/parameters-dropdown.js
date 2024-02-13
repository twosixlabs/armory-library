import { storeToRefs } from 'pinia';
import { computed } from 'vue';
import { useMetricsSettings } from '../stores/metrics-settings.js';
import Button from './button.js';
import { ChevronDownIcon } from './icons.js';

export default {
    props: {
        parameters: Object,
    },
    components: {
        Button,
        ChevronDownIcon,
    },
    setup(props) {
        const settings = useMetricsSettings();
        const { showParameters } = storeToRefs(settings);
        const availableParameters = computed(
            () => [...props.parameters].filter(
                (p) => !showParameters.value.includes(p)
            )
        );
        return { availableParameters, toggleParameter: settings.toggleParameter };
    },
    template: `
        <div class="dropdown">
            <Button :disabled="availableParameters.length == 0" tabindex="0">
                Parameters
                <ChevronDownIcon></ChevronDownIcon>
            </Button>
            <div
                tabindex="0"
                class="dropdown-content z-[1] shadow bg-base-100 rounded-box w-96 max-h-80 overflow-y-auto"
            >
                <ul class="menu p-2">
                    <li v-for="parameter in availableParameters" :key="parameter">
                        <a @click="toggleParameter(parameter)">
                            {{ parameter }}
                        </a>
                    </li>
                </ul>
            </div>
        </div>
    `,
};
