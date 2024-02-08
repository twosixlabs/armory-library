import { storeToRefs } from 'pinia';
import { ref } from 'vue';
import { useEvaluationData } from '../stores/evaluation-data.js';
import { useSelectedRuns } from '../stores/selected-runs.js';
import Heading from './heading.js';

export default {
    components: {
        Heading,
    },
    setup() {
        const evaluation = useEvaluationData();
        const selected = useSelectedRuns();
        const { runs: selectedRuns } = storeToRefs(selected);

        const selectAllRef = ref(null);
        selected.$subscribe((mutation, state) => {
            if (state.runs.length == 0) {
                selectAllRef.value.checked = false;
                selectAllRef.value.indeterminate = false;
            } else if (state.runs.length == evaluation.runs.length) {
                selectAllRef.value.checked = true;
                selectAllRef.value.indeterminate = false;
            } else {
                selectAllRef.value.checked = false;
                selectAllRef.value.indeterminate = true;
            }
        });

        const toggleSelectAll = () => {
            if (selected.runs.length < evaluation.runs.length) {
                selected.selectRuns(evaluation.runNames);
            } else {
                selected.selectRuns([]);
            }
        };

        return { evaluation, selected, selectedRuns, selectAllRef, toggleSelectAll };
    },
    template: `
        <div class="flex flex-col">
            <heading>Runs</heading>
            <div class="flex flex-row">
                <input
                    @change.prevent="toggleSelectAll"
                    ref="selectAllRef"
                    id="select-all"
                    type="checkbox"
                />
                <label for="select-all" class="ml-2">
                    Select all
                </label>
            </div>
            <div v-for="run in evaluation.runNames" :key="run" class="flex flex-row">
                <input
                    :id="run"
                    :checked="selectedRuns.includes(run)"
                    @change.prevent="selected.multiSelect(run)"
                    type="checkbox"
                />
                <label class="ml-2" @click="selected.singleSelect(run)">
                    {{ run }}
                </label>
            </div>
        </div>
    `
};
