import { storeToRefs } from 'pinia';
import { computed, ref } from 'vue';
import { RouterLink, useRouter } from 'vue-router';
import Button from '../components/button.js';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '../components/table.js';
import { useEvaluationData } from '../stores/evaluation-data.js';
import { useSelectedRuns } from '../stores/selected-runs.js';
import { humanizeDuration, humanizeTime } from '../utils/format.js';

export default {
    components: {
        Button,
        RouterLink,
        Table,
        TableBody,
        TableCell,
        TableHead,
        TableHeader,
        TableRow,
    },
    setup() {
        const router = useRouter();
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
                selected.selectRuns(evaluation.allRunIds);
            } else {
                selected.selectRuns([]);
            }
        };

        const multipleSelected = computed({
            get() {
                return selectedRuns.value.length > 1;
            },
        });

        const goToCompare = () => router.push({
            name: 'compare-runs-metrics',
            query: {
                runs: selected.runs,
            },
        });

        return {
            evaluation,
            goToCompare,
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
            <Table>
                <TableHead>
                    <tr>
                        <TableHeader class="rounded-tl-lg flex">
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
                        </TableHeader>
                        <TableHeader>Name</TableHeader>
                        <TableHeader>Started</TableHeader>
                        <TableHeader class="rounded-tr-lg">Duration</TableHeader>
                    </tr>
                </TableHead>
                <TableBody>
                    <TableRow v-for="run in evaluation.runs" :key="run.info.run_id">
                        <TableCell>
                            <input
                                :id="run.info.run_id"
                                :checked="selectedRuns.includes(run.info.run_id)"
                                @change.prevent="selected.multiSelect(run)"
                                class="hover:cursor-pointer"
                                type="checkbox"
                            />
                        </TableCell>
                        <TableCell>
                            <router-link
                                :to="{ name: 'single-run-metrics', params: { id: run.info.run_id } }"
                                class="hover:cursor-pointer text-twosix-blue"
                            >
                                {{ run.info.run_name }}
                            </router-link>
                        </TableCell>
                        <TableCell>
                            {{ run.info.start_time && humanizeTime(run.info.start_time) }}
                        </TableCell>
                        <TableCell>
                            {{ run.info.start_time && run.info.end_time && humanizeDuration(run.info.end_time - run.info.start_time) }}
                        </TableCell>
                    </TableRow>
                </TableBody>
            </Table>
            <div class="flex justify-end my-3">
                <Button
                    @click="goToCompare"
                    :disabled="!multipleSelected"
                >
                    Compare
                </Button>
            </div>
        </div>
    `,
};
