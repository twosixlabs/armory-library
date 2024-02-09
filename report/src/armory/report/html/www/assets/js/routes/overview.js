import { storeToRefs } from 'pinia';
import { computed, ref } from 'vue';
import Button from '../components/button.js';
import { useEvaluationData } from '../stores/evaluation-data.js';
import { useSelectedRuns } from '../stores/selected-runs.js';
import { humanizeDuration, humanizeTime } from '../utils/format.js';

const HeaderCell = {
    template: `
        <th scope="col" class="px-6 py-3">
            <slot></slot>
        </th>
    `,
};

const Row = {
    template: `
        <tr class="border-b even:bg-zinc-50">
            <slot></slot>
        </tr>
    `,
};

const Cell = {
    template: `
        <td class="px-6 py-4">
            <slot></slot>
        </td>
    `,
};

export default {
    components: {
        Button,
        Cell,
        HeaderCell,
        Row,
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

        const multipleSelected = computed({
            get() {
                return selectedRuns.value.length > 1;
            },
        });

        return {
            evaluation,
            humanizeDuration,
            humanizeTime,
            multipleSelected,
            selected,
            selectedRuns,
            selectAllRef,
            toggleSelectAll,
        };
    },
    template: `
        <div class="flex flex-col flex-grow mx-3">
            <table class="w-full text-sm text-left">
                <thead class="text-xs uppercase bg-zinc-200">
                    <tr>
                        <header-cell class="rounded-tl-lg flex">
                            <input
                                @change.prevent="toggleSelectAll"
                                ref="selectAllRef"
                                class="self-start"
                                id="select-all"
                                type="checkbox"
                            />
                            <label
                                class="hover:cursor-pointer ml-2"
                                for="select-all"
                            >
                                Select all
                            </label>
                        </header-cell>
                        <header-cell>Name</header-cell>
                        <header-cell>Started</header-cell>
                        <header-cell class="rounded-tr-lg">Duration</header-cell>
                    </tr>
                </thead>
                <tbody>
                    <row v-for="run in evaluation.runs" :key="run.info.run_name">
                        <cell>
                            <input
                                :id="run.info.run_name"
                                :checked="selectedRuns.includes(run.info.run_name)"
                                @change.prevent="selected.multiSelect(run)"
                                class="hover:cursor-pointer"
                                type="checkbox"
                            />
                        </cell>
                        <cell>
                            <a
                                :href="'?route=run&run=' + run.info.run_name"
                                class="hover:cursor-pointer text-twosix-blue"
                            >
                                {{ run.info.run_name }}
                            </a>
                        </cell>
                        <cell>
                            {{ run.info.start_time && humanizeTime(run.info.start_time) }}
                        </cell>
                        <cell>
                            {{ run.info.start_time && run.info.end_time && humanizeDuration(run.info.end_time - run.info.start_time) }}
                        </cell>
                    </row>
                </tbody>
            </table>
            <div class="flex justify-end my-3">
                <Button :disabled="!multipleSelected">Compare</Button>
            </div>
        </div>
    `,
};
