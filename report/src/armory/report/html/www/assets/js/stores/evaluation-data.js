import { defineStore } from 'pinia';
import armoryEvaluationData from 'armory-evaluation-data';

export const useEvaluationData = defineStore('evaluation-data', {
    state: () => armoryEvaluationData,
    getters: {
        allRunIds(state) {
            return state.runs.map((run) => run.info.run_id);
        },
    },
});
