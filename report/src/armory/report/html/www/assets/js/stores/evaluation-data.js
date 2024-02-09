import { defineStore } from 'pinia';
import armoryEvaluationData from 'armory-evaluation-data';
import { useSelectedRuns } from './selected-runs.js';

export const useEvaluationData = defineStore('evaluation-data', {
    state: () => armoryEvaluationData,
    getters: {
        allRunIds(state) {
            return state.runs.map((run) => run.info.run_id);
        },
        runNames(state) {
            return state.runs.map((run) => run.info.run_name);
        },
        selectedRuns(state) {
            const selected = useSelectedRuns();
            return state.runs.filter((run) => selected.runs.includes(run.info.run_name));
        }
    },
});
