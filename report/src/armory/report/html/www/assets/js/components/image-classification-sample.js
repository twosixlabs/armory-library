import { computed } from 'vue';

export default {
    props: {
        batch: Number,
        lhsChain: String,
        rhsChain: String,
        run: Object,
        sample: Number,
    },
    setup(props) {
        const lhsImage = computed(() => {
            const file = props.run.artifacts[props.lhsChain]?.[props.batch]?.[props.sample];
            if (file) {
                return `./assets/img/${props.run.info.run_id}/${file}`;
            }
            return "";
        });

        const rhsImage = computed(() => {
            const file = props.run.artifacts[props.rhsChain]?.[props.batch]?.[props.sample];
            if (file) {
                return `./assets/img/${props.run.info.run_id}/${file}`;
            }
            return "";
        });

        return { lhsImage, rhsImage };
    },
    template: `
        <div v-if="lhsImage && rhsImage" class="flex gap-2 items-center justify-center">
            <div>Predicted Label: 38</div>
            <div class="diff aspect-square max-w-xl">
                <div class="diff-item-1">
                    <img :src="rhsImage" />
                </div>
                <div class="diff-item-2">
                    <img :src="lhsImage" />
                </div>
                <div class="diff-resizer"></div>
            </div>
            <div>Predicted Label: 37</div>
        </div>
    `,
};
