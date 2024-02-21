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

    const createSetting = (key, defaultKey) => computed({
        get() {
            const value = route.value.query[key];
            if (value == undefined && defaultKey) {
                return evaluationData.settings[defaultKey];
            }
            return value || "";
        },
        set(value) {
            updateQuery({ [key]: value });
        },
    });

    const lhsChain = createSetting("lhsChain", "lhs_chain");
    const rhsChain = createSetting("rhsChain", "rhs_chain");
    const batch = createSetting("batch", "batch");
    const sample = createSetting("sample", "sample");

    const sideBySide = computed({
        get() {
            const sideBySide = route.value.query.sideBySide;
            if (sideBySide == undefined) {
                return false;
            }
            return sideBySide == "true";
        },
        set(sideBySide) {
            updateQuery({ sideBySide });
        },
    });

    return {
        lhsChain,
        rhsChain,
        batch,
        sample,
        sideBySide,
    };
});
