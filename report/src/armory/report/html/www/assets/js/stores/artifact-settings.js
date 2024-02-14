import { defineStore } from 'pinia';
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useEvaluationData } from './evaluation-data.js';

export const useArtifactSettings = defineStore('artifact-settings', () => {
    const evaluationData = useEvaluationData();

    const router = useRouter();
    const route = router.currentRoute;
    const updateQuery = (query) => {
        router.push({ query: { ...route.value.query, ...query } }, { replace: true });
    };

    const createSetting = (key) => computed({
        get() {
            return route.value.query[key] || "";
        },
        set(value) {
            updateQuery({ [key]: value });
        },
    });

    const lhsChain = createSetting("lhsChain");
    const rhsChain = createSetting("rhsChain");
    const batch = createSetting("batch");
    const sample = createSetting("sample");

    return {
        lhsChain,
        rhsChain,
        batch,
        sample,
    };
});
