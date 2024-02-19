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

        const selectedRuns = ref([...evaluation.allRunIds]);
        const selectAll = computed({
            get() {
                return selectedRuns.value.length == evaluation.runs.length;
            },
            set(select) {
                if (select) {
                    selectedRuns.value = [...evaluation.allRunIds];
                } else {
                    selectedRuns.value = [];
                }
            }
        });
        const partialSelection = computed(() => 
            selectedRuns.value.length > 0 && selectedRuns.value.length < evaluation.runs.length
        );
        const multipleSelected = computed(() => selectedRuns.value.length > 1);

        const goToCompare = () => router.push({
            name: 'compare-runs-metrics',
            query: {
                runs: selectedRuns.value,
            },
        });

        return {
            evaluation,
            goToCompare,
            humanizeDuration,
            humanizeTime,
            multipleSelected,
            partialSelection,
            selectedRuns,
            selectAll,
        };
    },
    template: `
        <div class="flex flex-col flex-grow mx-3">
            <Table>
                <TableHead>
                    <tr>
                        <TableHeader class="rounded-tl-lg flex">
                            <input
                                v-model="selectAll"
                                :indeterminate.prop="partialSelection"
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
                        <TableHeader>Duration</TableHeader>
                        <TableHeader class="rounded-tr-lg">Status</TableHeader>
                    </tr>
                </TableHead>
                <TableBody>
                    <TableRow v-for="run in evaluation.runs" :key="run.info.run_id">
                        <TableCell>
                            <input
                                v-model="selectedRuns"
                                :id="run.info.run_id"
                                :value="run.info.run_id"
                                class="hover:cursor-pointer"
                                type="checkbox"
                            />
                        </TableCell>
                        <TableCell>
                            <router-link
                                :to="{ name: 'single-run', params: { id: run.info.run_id } }"
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
                        <TableCell>
                            {{ run.info.status }}
                        </TableCell>
                    </TableRow>
                </TableBody>
            </Table>
            <div class="my-3">
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
